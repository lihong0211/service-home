import hashlib
import os
import tempfile
from fastapi import Request, UploadFile
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dashscope import TextEmbedding

from config.ai import DEFAULT_EMBEDDING_MODEL
from service.ai._dashscope_common import stream_dashscope_sse

QDRANT_PATH = os.path.join(os.path.dirname(__file__), "../../../data/vector_dbs")
EMBED_DIM = 1536


def _get_qdrant():
    return QdrantClient(path=QDRANT_PATH)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    all_vecs = []
    for i in range(0, len(texts), 10):
        batch = texts[i:i + 10]
        resp = TextEmbedding.call(model=DEFAULT_EMBEDDING_MODEL, input=batch)
        all_vecs.extend([e["embedding"] for e in resp.output["embeddings"]])
    return all_vecs


async def pdf_index_api(request: Request):
    form = await request.form()
    pdf_file: UploadFile = form.get("pdf")
    if not pdf_file:
        return {"code": 400, "msg": "请上传 PDF 文件"}

    content = await pdf_file.read()
    file_hash = hashlib.md5(content).hexdigest()[:12]
    collection_name = f"pdf_{file_hash}"

    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        pages_text = [p.extract_text() or "" for p in reader.pages]
        page_count = len(pages_text)
        full_text = "\n".join(pages_text)
    except Exception as e:
        return {"code": 500, "msg": f"PDF 解析失败: {str(e)}"}

    # chunk by 800 chars
    chunks = []
    for i in range(0, len(full_text), 800):
        chunk = full_text[i:i + 800].strip()
        if chunk:
            chunks.append(chunk)

    if not chunks:
        return {"code": 400, "msg": "PDF 内容为空"}

    client = _get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        vecs = _embed_texts(chunks)
        points = [PointStruct(id=i, vector=vecs[i], payload={"text": chunks[i], "chunk_idx": i}) for i in range(len(chunks))]
        client.upsert(collection_name=collection_name, points=points)

    return {
        "code": 0, "msg": "success",
        "data": {"index_id": collection_name, "page_count": page_count, "chunk_count": len(chunks)},
    }


import io  # noqa: E402 - needed for BytesIO


async def pdf_ask_api(request: Request):
    body = await request.json()
    index_id = body.get("index_id", "")
    question = body.get("question", "")

    if not index_id or not question:
        return {"code": 400, "msg": "缺少参数"}

    client = _get_qdrant()
    q_vec = _embed_texts([question])[0]
    results = client.query_points(
        collection_name=index_id,
        query=q_vec,
        limit=5,
    )
    context = "\n\n".join(p.payload["text"] for p in results.points)

    system_prompt = "你是一个 PDF 文档助手，只根据提供的文档内容回答问题。如果文档中没有相关信息，请明确说明。"
    user_prompt = f"文档内容：\n{context}\n\n问题：{question}"
    return StreamingResponse(
        stream_dashscope_sse(system_prompt, user_prompt), media_type="text/event-stream"
    )
