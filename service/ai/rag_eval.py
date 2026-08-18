"""
RAGAS 集成：给一次 RAG 问答（query + answer + retrieved contexts，可选 ground_truth）打分。

两类指标：
- 不需要 ground_truth 就能算：faithfulness（答案是否忠于检索到的上下文，有没有编造）、
  answer_relevancy（答案有没有正面回应问题）。
- 需要 ground_truth 才能算：context_precision（检索片段里真正相关的占比）、
  context_recall（回答问题需要的信息是否都召回了）、answer_correctness（跟标准答案的匹配度）。
  没传 ground_truth 时这三项直接跳过，不强算、不编造分数。

evaluator LLM/embeddings 复用项目已有的 DashScope（OpenAI 兼容）配置，不需要额外的 OpenAI key。
instructor（RAGAS 内部用它做结构化打分）依赖模型的 function calling 能力，DashScope 的
qwen-turbo 支持，实测可用；如果换成不支持 function calling 的模型，这里会直接报错。
"""
import asyncio
import logging
import random

import anyio
from fastapi import Request
from openai import AsyncOpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from config.ai import DASHSCOPE_BASE_URL, DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, dashscope_api_key
from service.ai.rag import rag_chat
from utils.http_body import query_dict, read_json_optional

logger = logging.getLogger(__name__)

_async_client: AsyncOpenAI | None = None
_llm = None
_embeddings = None
_zh_prompts_by_synthesizer: dict = {}


def _get_evaluator():
    """惰性初始化 evaluator LLM/embeddings，模块内只建一次，跨请求复用同一个 client。"""
    global _async_client, _llm, _embeddings
    if _llm is None:
        _async_client = AsyncOpenAI(api_key=dashscope_api_key(), base_url=DASHSCOPE_BASE_URL)
        _llm = llm_factory(DEFAULT_CHAT_MODEL, client=_async_client)
        _embeddings = embedding_factory(
            "openai", model=DEFAULT_EMBEDDING_MODEL, client=_async_client, interface="modern"
        )
    return _llm, _embeddings


async def _get_zh_prompts_by_synthesizer(llm) -> dict:
    """
    默认 query synthesizer（single_hop_specific / multi_hop_abstract / multi_hop_specific）
    的 prompt 模板自带英文 few-shot 示例，即便源文档是中文也会把输出带偏成英文，光加一句
    中文提示词（llm_context）压不住。用 adapt_prompts 把模板连示例一起翻译成中文，按类名
    缓存，跨请求复用，避免每次都重新翻译。

    刻意缓存"翻译后的 prompts"而不是缓存 synthesizer 实例/distribution 本身——
    default_query_distribution(llm, kg) 会按每次请求实际构建出的知识图谱是否存在可用的
    实体关系簇（cluster）来决定是否启用 multi_hop 系列，chunk 数少或语义稀疏时经常没有
    可用簇；如果直接复用一份写死的 distribution，等于绕过了这个按 kg 过滤的逻辑，会导致
    multi_hop synthesizer 在没有簇的知识图谱上报错
    "No relationships match the provided condition. Cannot form clusters."（已实测触发）。
    """
    global _zh_prompts_by_synthesizer
    if not _zh_prompts_by_synthesizer:
        from ragas.testset.synthesizers import default_query_distribution

        for synthesizer, _weight in default_query_distribution(llm):
            adapted = await synthesizer.adapt_prompts("chinese", llm=llm)
            _zh_prompts_by_synthesizer[type(synthesizer).__name__] = adapted
    return _zh_prompts_by_synthesizer


async def evaluate_sample(
    query: str, answer: str, contexts: list[str], ground_truth: str | None = None
) -> dict:
    """
    对一组 (query, answer, contexts[, ground_truth]) 跑 RAGAS 打分。
    每个指标独立 try/except：单个指标算失败（比如 LLM 结构化输出解析失败）不拖累其他指标，
    失败的那项在结果里是 null + 对应的 error 说明，不是让整个请求 500。
    """
    llm, embeddings = _get_evaluator()

    async def _score(name: str, run):
        """run 是个零参 async 可调用（而不是提前建好的 coroutine 对象）：连"调用 .ascore(...)
        本身传错参数"这种构造期报错都要能被这里的 try/except 兜住，不能让它在 gather 之前
        就直接抛出去，那样会导致同一批里其他能算的指标也算不出来。"""
        try:
            result = await run()
            return name, round(float(result.value), 4), None
        except Exception as e:  # noqa: BLE001 - 逐指标兜底，不让一个指标的失败拖垮整体
            logger.exception("RAGAS 指标 %s 计算失败", name)
            return name, None, str(e)

    tasks = [
        _score("faithfulness", lambda: Faithfulness(llm=llm).ascore(
            user_input=query, response=answer, retrieved_contexts=contexts
        )),
        _score("answer_relevancy", lambda: AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(
            user_input=query, response=answer
        )),
    ]
    if ground_truth:
        tasks += [
            _score("context_precision", lambda: ContextPrecision(llm=llm).ascore(
                user_input=query, reference=ground_truth, retrieved_contexts=contexts
            )),
            _score("context_recall", lambda: ContextRecall(llm=llm).ascore(
                user_input=query, reference=ground_truth, retrieved_contexts=contexts
            )),
            _score("answer_correctness", lambda: AnswerCorrectness(llm=llm, embeddings=embeddings).ascore(
                user_input=query, response=answer, reference=ground_truth
            )),
        ]

    results = await asyncio.gather(*tasks)
    scores: dict = {}
    errors: dict = {}
    for name, value, error in results:
        scores[name] = value
        if error:
            errors[name] = error
    if errors:
        scores["_errors"] = errors
    return scores


def evaluate_rag_api(request: Request):
    """
    POST /ai/rag/evaluate
    body:
      - kb_id / kb_name, question：跑一次真实 RAG（复用 rag.rag_chat），用它的 answer/检索片段来打分
      - answer, contexts（list[str]）：如果已经有现成的问答对，直接传这两项可以跳过重新跑 RAG
      - ground_truth（可选）：给了才会算 context_precision/context_recall/answer_correctness
      - top_k（可选，默认5）
    返回：{question, answer, contexts, ground_truth, scores}
    """
    data = anyio.from_thread.run(read_json_optional, request) or {}
    question = (data.get("question") or data.get("query") or "").strip()
    if not question:
        raise ValueError("请提供 question")
    ground_truth = (data.get("ground_truth") or "").strip() or None

    answer = data.get("answer")
    contexts = data.get("contexts")

    if answer and contexts:
        # 已经有现成问答对，直接评测，不重新跑 RAG
        answer = str(answer)
        contexts = [str(c) for c in contexts if str(c).strip()]
    else:
        kb_id = data.get("kb_id")
        kb_name = (data.get("kb_name") or "").strip()
        if not kb_id and not kb_name:
            raise ValueError("请提供 kb_id 或 kb_name（或者直接传 answer + contexts 跳过重新跑 RAG）")
        if kb_id is not None:
            try:
                kb_id = int(kb_id)
            except (TypeError, ValueError):
                raise ValueError("kb_id 必须为数字")
        top_k = data.get("top_k", 5)
        try:
            top_k = max(1, min(20, int(top_k)))
        except (TypeError, ValueError):
            top_k = 5
        rag_result = rag_chat(kb_id=kb_id, kb_name=kb_name or None, question=question, top_k=top_k)
        answer = rag_result.get("answer", "")
        contexts = rag_result.get("full_contexts") or []

    if not contexts:
        raise ValueError("没有检索到任何上下文，无法评测（知识库为空或该问题检索不到相关内容）")

    scores = anyio.from_thread.run(evaluate_sample, question, answer, contexts, ground_truth)

    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "scores": scores,
        },
    }


# ---------------------------------------------------------------------------
# 测试集自动生成：喂知识库分段给 RAGAS，生成「问题-上下文-标准答案」三元组落库，
# 供反复做回归评测——不是生成一次看一眼就扔，改了 prompt/模型/检索参数后应该拿同一批
# 题重新跑一遍看有没有退步。生成出来的 ground_truth 是模型合成的，不是人工核对过的
# 金标准，适合当"能跑起来的起点"，正式当团队基准前最好抽几条人工过一遍。
# ---------------------------------------------------------------------------

# 喂给生成器的分段数上限：生成阶段要对每个分段跑 Summary/Themes/NER 等好几轮 LLM 抽取，
# 分段数一多耗时会指数级上升——实测 30 个分段生成 3 条题目跑了 3 分钟以上还没完成，
# 10 个分段生成同样数量在 1~2 分钟内能跑完，取这个更适合交互式使用的量级。
# 超过上限时随机采样而不是取前 N 个，避免只覆盖文档开头。
MAX_TESTSET_CHUNKS = 10


def _fetch_kb_chunks(kb_id: int, limit: int = MAX_TESTSET_CHUNKS) -> list[str]:
    """取一个知识库全部分段的原文文本，超过 limit 时随机采样。"""
    from model.ai import KnowledgeBaseDocument, KnowledgeBaseSegment

    docs = KnowledgeBaseDocument.select_by({"knowledge_base_id": kb_id})
    doc_ids = [d.id for d in docs]
    if not doc_ids:
        return []
    segments = KnowledgeBaseSegment.select_by({"document_id": doc_ids})
    texts = [s.text for s in segments if (s.text or "").strip()]
    if len(texts) > limit:
        texts = random.sample(texts, limit)
    return texts


def _generate_testset(kb_id: int, size: int) -> list[dict]:
    """
    调 RAGAS TestsetGenerator 生成测试集。
    刻意写成同步函数、直接调用，不包一层 async/anyio.from_thread.run——
    generate/apply_transforms 底层会自己起一个 event loop 跑里面一堆异步
    LLM 调用；如果从已经在跑的 event loop（anyio.from_thread.run 桥接过去的那个）里调它，
    会触发"loop 套 loop"死锁，实测直接卡死不报错，很隐蔽。跟 rag_chat 一样直接同步调用即可。
    输出列名（实测确认，不是文档直接抄的——RAGAS 各版本字段名变过）：
    user_input=问题，reference=标准答案，reference_contexts=生成时依据的原文片段列表，
    synthesizer_name=生成方式（single_hop_specific/multi_hop_abstract 等，multi_hop 的题
    需要综合多个片段才能回答，比 single_hop 更难）。
    默认 query synthesizer 的 prompt 模板自带英文 few-shot 示例，即便源文档是中文，模型也
    会偏向生成英文问答——用 _get_zh_prompts_by_synthesizer 把模板连示例一起翻译成中文再用。
    这里没有直接调 generate_with_chunks（它内部会用没翻译过的默认 distribution），而是手动
    重复它构建知识图谱、跑 transforms 那部分逻辑，为的是能拿到构建好的知识图谱去调
    default_query_distribution(llm, kg) ——只有这样 multi_hop synthesizer 才会按这个知识库
    实际有没有可用的实体关系簇来决定要不要启用，跟 generate_with_chunks 内部行为保持一致。
    """
    from langchain_core.documents import Document
    from ragas.run_config import RunConfig
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.synthesizers import default_query_distribution
    from ragas.testset.synthesizers.generate import TestsetGenerator
    from ragas.testset.transforms import apply_transforms, default_transforms_for_prechunked

    texts = _fetch_kb_chunks(kb_id)
    if not texts:
        raise ValueError("该知识库没有可用的分段文本，请先完成文档解析和向量化")
    llm, embeddings = _get_evaluator()
    zh_prompts = asyncio.run(_get_zh_prompts_by_synthesizer(llm))

    nodes = [
        Node(type=NodeType.CHUNK, properties={"page_content": t, "document_metadata": {}})
        for t in texts
        if t.strip()
    ]
    kg = KnowledgeGraph(nodes=nodes)
    apply_transforms(
        kg, default_transforms_for_prechunked(llm=llm, embedding_model=embeddings), run_config=RunConfig()
    )

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings, knowledge_graph=kg)
    query_distribution = default_query_distribution(llm, kg)
    for synthesizer, _weight in query_distribution:
        prompts = zh_prompts.get(type(synthesizer).__name__)
        if prompts:
            synthesizer.set_prompts(**prompts)
        synthesizer.llm_context = "请用简体中文生成问题和参考答案，不要出现英文。"

    testset = generator.generate(testset_size=size, query_distribution=query_distribution)
    df = testset.to_pandas()
    items = []
    for _, row in df.iterrows():
        question = str(row.get("user_input") or "").strip()
        ground_truth = str(row.get("reference") or "").strip()
        if not question or not ground_truth:
            continue
        contexts = row.get("reference_contexts") or []
        items.append({
            "question": question,
            "ground_truth": ground_truth,
            "source_context": "\n\n---\n\n".join(str(c) for c in contexts)[:5000],
            "synthesizer": str(row.get("synthesizer_name") or ""),
        })
    return items


def generate_testset_api(request: Request):
    """
    POST /ai/rag/testset/generate  body: {kb_id, size?(默认10，上限20)}
    生成结果落库（rag_testset_item）。这是个慢接口——前台跑（比如直接 python main.py）
    10 个分段生成几条题目约 1~2 分钟；同样的请求打到本机 launchd 常驻的后台服务
    （com.lihong.service-home，走 launchctl 管理）实测慢了一个数量级以上（单次 LLM 调用
    观测到 240s+），大概率是 macOS 对常驻后台进程的资源节流（App Nap 一类机制）——这类
    长时间高频调用外部 API 的批量任务，不适合扔给这种后台常驻服务跑。前端要给够超时。
    """
    data = anyio.from_thread.run(read_json_optional, request) or {}
    kb_id = data.get("kb_id")
    if not kb_id:
        raise ValueError("请提供 kb_id")
    try:
        kb_id = int(kb_id)
    except (TypeError, ValueError):
        raise ValueError("kb_id 必须为数字")
    size = data.get("size", 10)
    try:
        size = max(1, min(20, int(size)))
    except (TypeError, ValueError):
        size = 10

    from model.ai import KnowledgeBase
    kb = KnowledgeBase.get_by_id(kb_id)
    kb_name = kb.name if kb else None

    items = _generate_testset(kb_id, size)
    if not items:
        raise ValueError("生成失败：RAGAS 没有产出任何有效的问答对，换一个知识库或稍后重试")

    from app.database import SessionLocal
    from model.ai.rag_testset import RagTestsetItem

    session = SessionLocal()
    saved = []
    try:
        for it in items:
            row = RagTestsetItem(
                kb_id=kb_id,
                kb_name=kb_name,
                question=it["question"],
                ground_truth=it["ground_truth"],
                source_context=it["source_context"],
                synthesizer=it["synthesizer"],
            )
            session.add(row)
            session.flush()
            saved.append({"id": row.id, "kb_id": kb_id, "kb_name": kb_name, **it})
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return {"code": 0, "msg": "ok", "data": {"items": saved, "total": len(saved)}}


def list_testset_api(request: Request):
    """GET /ai/rag/testset/list?kb_id=21  列出某个知识库已生成、落库的测试集。"""
    q = query_dict(request)
    kb_id = q.get("kb_id")
    if not kb_id:
        raise ValueError("请提供 kb_id")

    from app.database import SessionLocal
    from model.ai.rag_testset import RagTestsetItem

    session = SessionLocal()
    try:
        rows = (
            session.query(RagTestsetItem)
            .filter(RagTestsetItem.kb_id == int(kb_id))
            .order_by(RagTestsetItem.id.desc())
            .all()
        )
        items = [
            {
                "id": r.id,
                "kb_id": r.kb_id,
                "kb_name": r.kb_name,
                "question": r.question,
                "ground_truth": r.ground_truth,
                "source_context": r.source_context,
                "synthesizer": r.synthesizer,
                "created_at": r.create_at.isoformat() if r.create_at else None,
            }
            for r in rows
        ]
        return {"code": 0, "msg": "ok", "data": {"items": items, "total": len(items)}}
    finally:
        session.close()


def _testset_item_to_dict(r) -> dict:
    return {
        "id": r.id,
        "kb_id": r.kb_id,
        "kb_name": r.kb_name,
        "question": r.question,
        "ground_truth": r.ground_truth,
        "source_context": r.source_context,
        "synthesizer": r.synthesizer,
        "created_at": r.create_at.isoformat() if r.create_at else None,
    }


def create_testset_item_api(request: Request):
    """
    POST /ai/rag/testset/create  body: {kb_id, question, ground_truth, source_context?}
    手动补一条测试集数据（不经过 RAGAS 生成），synthesizer 固定标成 manual，跟自动生成的
    区分开，批量评测/展示逻辑不用改。
    """
    data = anyio.from_thread.run(read_json_optional, request) or {}
    kb_id = data.get("kb_id")
    if not kb_id:
        raise ValueError("请提供 kb_id")
    try:
        kb_id = int(kb_id)
    except (TypeError, ValueError):
        raise ValueError("kb_id 必须为数字")
    question = (data.get("question") or "").strip()
    ground_truth = (data.get("ground_truth") or "").strip()
    if not question or not ground_truth:
        raise ValueError("请提供 question 和 ground_truth")
    source_context = (data.get("source_context") or "").strip() or None

    from app.database import SessionLocal
    from model.ai import KnowledgeBase
    from model.ai.rag_testset import RagTestsetItem

    kb = KnowledgeBase.get_by_id(kb_id)
    kb_name = kb.name if kb else None

    session = SessionLocal()
    try:
        row = RagTestsetItem(
            kb_id=kb_id,
            kb_name=kb_name,
            question=question,
            ground_truth=ground_truth,
            source_context=source_context,
            synthesizer="manual",
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return {"code": 0, "msg": "ok", "data": _testset_item_to_dict(row)}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def update_testset_item_api(request: Request, item_id: int):
    """
    PUT /ai/rag/testset/{item_id}  body: {question?, ground_truth?, source_context?}
    只改传了的字段，没传的保持原样——用来人工修正 RAGAS 生成的问答对。
    """
    data = anyio.from_thread.run(read_json_optional, request) or {}

    from app.database import SessionLocal
    from model.ai.rag_testset import RagTestsetItem

    session = SessionLocal()
    try:
        row = session.query(RagTestsetItem).filter(RagTestsetItem.id == item_id).first()
        if not row:
            raise FileNotFoundError("测试集数据不存在")

        if "question" in data:
            question = (data.get("question") or "").strip()
            if not question:
                raise ValueError("question 不能为空")
            row.question = question
        if "ground_truth" in data:
            ground_truth = (data.get("ground_truth") or "").strip()
            if not ground_truth:
                raise ValueError("ground_truth 不能为空")
            row.ground_truth = ground_truth
        if "source_context" in data:
            row.source_context = (data.get("source_context") or "").strip() or None

        session.commit()
        session.refresh(row)
        return {"code": 0, "msg": "ok", "data": _testset_item_to_dict(row)}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def delete_testset_item_api(request: Request, item_id: int):
    """DELETE /ai/rag/testset/{item_id}  硬删——测试集数据没有下游引用，不用软删。"""
    from app.database import SessionLocal
    from model.ai.rag_testset import RagTestsetItem

    session = SessionLocal()
    try:
        row = session.query(RagTestsetItem).filter(RagTestsetItem.id == item_id).first()
        if not row:
            raise FileNotFoundError("测试集数据不存在")
        session.delete(row)
        session.commit()
        return {"code": 0, "msg": "ok", "data": {"id": item_id}}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


_BATCH_METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]


async def _batch_evaluate_async(kb_id: int, items: list[dict]) -> list[dict]:
    """
    对一批测试集问题依次跑真实 RAG + RAGAS 打分。故意串行而不是并发——每条本身就要
    好几次 LLM 调用，并发太高容易触发限流，这里用耗时换稳定；单条失败不影响其余条目。
    """
    results = []
    for it in items:
        try:
            rag_result = rag_chat(kb_id=kb_id, question=it["question"], top_k=5)
            answer = rag_result.get("answer", "")
            contexts = rag_result.get("full_contexts") or []
            scores = await evaluate_sample(it["question"], answer, contexts, it["ground_truth"])
            results.append({"id": it["id"], "question": it["question"], "answer": answer, "scores": scores, "error": None})
        except Exception as e:  # noqa: BLE001 - 单条失败要记录，不能让整批中断
            logger.exception("批量评测单条失败: testset_id=%s", it.get("id"))
            results.append({"id": it["id"], "question": it["question"], "answer": None, "scores": {}, "error": str(e)})
    return results


def batch_evaluate_api(request: Request):
    """
    POST /ai/rag/testset/evaluate  body: {kb_id, testsetIds?: number[]}
    testsetIds 不传则跑该知识库全部测试集。对整套测试集依次跑一遍真实 RAG + RAGAS，
    返回逐条结果 + 五项指标的平均分（跳过算失败的 null，不当 0 拉低平均），用来看这次
    改动有没有整体退步。慢接口：题目数 × 单条耗时（~15-20s），调用方要给足够长的超时。
    """
    data = anyio.from_thread.run(read_json_optional, request) or {}
    kb_id = data.get("kb_id")
    if not kb_id:
        raise ValueError("请提供 kb_id")
    kb_id = int(kb_id)
    testset_ids = data.get("testsetIds") or data.get("testset_ids")

    from app.database import SessionLocal
    from model.ai.rag_testset import RagTestsetItem

    session = SessionLocal()
    try:
        query = session.query(RagTestsetItem).filter(RagTestsetItem.kb_id == kb_id)
        if testset_ids:
            query = query.filter(RagTestsetItem.id.in_([int(i) for i in testset_ids]))
        rows = query.all()
        items = [{"id": r.id, "question": r.question, "ground_truth": r.ground_truth} for r in rows]
    finally:
        session.close()

    if not items:
        raise ValueError("该知识库没有可用的测试集，请先生成测试集")

    results = anyio.from_thread.run(_batch_evaluate_async, kb_id, items)

    # avg 看整体水平，min 看有没有拖后腿的单条——平均分再高，只要有一条烂得离谱也说明
    # 某类问题没处理好，光看平均容易被掩盖。
    summary_avg = {}
    summary_min = {}
    for m in _BATCH_METRIC_NAMES:
        values = [r["scores"].get(m) for r in results if r["scores"].get(m) is not None]
        summary_avg[m] = round(sum(values) / len(values), 4) if values else None
        summary_min[m] = round(min(values), 4) if values else None

    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "results": results,
            "summary": summary_avg,
            "summaryMin": summary_min,
            "total": len(results),
        },
    }
