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
from utils.http_body import read_json_optional

logger = logging.getLogger(__name__)

_async_client: AsyncOpenAI | None = None
_llm = None
_embeddings = None


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
