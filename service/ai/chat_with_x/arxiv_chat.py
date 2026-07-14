import io
import re
import requests
from fastapi import Request
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dashscope import TextEmbedding
import os

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


def _parse_arxiv_id(arxiv_input: str) -> str:
    arxiv_input = arxiv_input.strip()
    m = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", arxiv_input)
    if m:
        return m.group(1)
    return arxiv_input


async def arxiv_index_api(request: Request):
    body = await request.json()
    arxiv_input = body.get("arxiv_id", "")
    if not arxiv_input:
        return {"code": 400, "msg": "请输入 ArXiv ID 或 URL"}

    arxiv_id = _parse_arxiv_id(arxiv_input)
    clean_id = arxiv_id.replace(".", "_").replace("/", "_")
    collection_name = f"arxiv_{clean_id}"

    # Get metadata
    meta_url = f"https://export.arxiv.org/abs/{arxiv_id}"
    title = arxiv_id
    abstract = ""
    try:
        r = requests.get(meta_url, timeout=15)
        title_m = re.search(r'<h1 class="title[^"]*">\s*<span[^>]*>Title:</span>\s*(.*?)</h1>', r.text, re.S)
        if title_m:
            title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip()
        abs_m = re.search(r'<blockquote class="abstract[^"]*">\s*<span[^>]*>Abstract:</span>\s*(.*?)</blockquote>', r.text, re.S)
        if abs_m:
            abstract = re.sub(r"<[^>]+>", "", abs_m.group(1)).strip()
    except Exception:
        pass

    # Download PDF
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    client = _get_qdrant()
    existing = [c.name for c in client.get_collections().collections]

    if collection_name not in existing:
        try:
            r = requests.get(pdf_url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            return {"code": 500, "msg": f"下载 ArXiv PDF 失败: {str(e)}"}

        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(r.content))
        pages_text = [p.extract_text() or "" for p in reader.pages]
        page_count = len(pages_text)
        full_text = "\n".join(pages_text)

        chunks = [full_text[i:i + 800].strip() for i in range(0, len(full_text), 800) if full_text[i:i + 800].strip()]
        if not chunks:
            return {"code": 400, "msg": "论文内容为空"}

        client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE))
        vecs = _embed_texts(chunks)
        points = [PointStruct(id=i, vector=vecs[i], payload={"text": chunks[i]}) for i in range(len(chunks))]
        client.upsert(collection_name=collection_name, points=points)
    else:
        page_count = 0

    return {
        "code": 0, "msg": "success",
        "data": {"index_id": collection_name, "arxiv_id": arxiv_id, "title": title, "abstract": abstract, "page_count": page_count},
    }


async def arxiv_ask_api(request: Request):
    body = await request.json()
    index_id = body.get("index_id", "")
    question = body.get("question", "")

    if not index_id or not question:
        return {"code": 400, "msg": "缺少参数"}

    client = _get_qdrant()
    q_vec = _embed_texts([question])[0]
    results = client.query_points(collection_name=index_id, query=q_vec, limit=5)
    context = "\n\n".join(p.payload["text"] for p in results.points)

    system_prompt = "你是一个学术论文助手，帮助用户理解 ArXiv 论文内容。根据提供的论文片段回答问题，如有必要用中文解释。"
    user_prompt = f"论文内容：\n{context}\n\n问题：{question}"
    return StreamingResponse(
        stream_dashscope_sse(system_prompt, user_prompt), media_type="text/event-stream"
    )
