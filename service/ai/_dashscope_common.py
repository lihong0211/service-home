# service/ai/_dashscope_common.py
# 收敛 starter_agents/llm_apps/chat_with_x/advanced_agents 里重复的
# 「DashScope client 初始化 + 流式 SSE 输出」样板代码。
import json
import logging
import time

from dashscope import Generation
from openai import OpenAI

from config.ai import DASHSCOPE_BASE_URL, DEFAULT_CHAT_MODEL, dashscope_api_key

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 2


def get_dashscope_client(timeout: float = REQUEST_TIMEOUT_SECONDS) -> OpenAI:
    """OpenAI 兼容模式的 DashScope client（用于 chat.completions / embeddings）。"""
    return OpenAI(api_key=dashscope_api_key(), base_url=DASHSCOPE_BASE_URL, timeout=timeout)


def call_generation_with_retry(*, model: str, messages: list[dict], **kwargs):
    """带超时 + 重试的 Generation.call 包装。stream=True 时仅重试「发起调用」这一步，
    一旦开始产出 chunk 就不再重试（避免向调用方重复吐出内容）。"""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return Generation.call(
                model=model,
                messages=messages,
                api_key=dashscope_api_key(),
                request_timeout=REQUEST_TIMEOUT_SECONDS,
                **kwargs,
            )
        except Exception as e:  # noqa: BLE001 - 需要捕获 SDK 抛出的各种异常做重试
            last_exc = e
            if attempt < MAX_RETRIES:
                logger.warning("DashScope Generation.call 失败，第 %d 次重试: %s", attempt + 1, e)
                time.sleep(2**attempt)
    assert last_exc is not None
    raise last_exc


def call_openai_chat_with_retry(client, *, model: str, messages: list[dict], **kwargs):
    """OpenAI 兼容 client 的 chat.completions.create 重试包装（用于 RAG 等直接持有 client 的调用方）。"""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < MAX_RETRIES:
                logger.warning("OpenAI chat.completions.create 失败，第 %d 次重试: %s", attempt + 1, e)
                time.sleep(2**attempt)
    assert last_exc is not None
    raise last_exc


def _response_error(resp) -> str:
    """DashScope 出错时 output 为 None，错误信息在 code/message 字段（不抛异常，需要显式检查）。"""
    return f"{getattr(resp, 'code', '')}: {getattr(resp, 'message', '')}".strip(": ")


def call_dashscope_text(prompt: str, model: str = DEFAULT_CHAT_MODEL, system_prompt: str | None = None) -> str:
    """非流式单轮调用，返回去除首尾空白的文本内容。"""
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
        {"role": "user", "content": prompt}
    ]
    resp = call_generation_with_retry(model=model, messages=messages, result_format="message")
    if resp.output is None:
        raise RuntimeError(f"DashScope 调用失败: {_response_error(resp)}")
    return resp.output.choices[0].message.content.strip()


def stream_dashscope_sse(system_prompt: str, user_prompt: str, model: str = DEFAULT_CHAT_MODEL):
    """生成器：调用 DashScope Generation 流式接口，逐段以 SSE 格式（{'response': delta}）yield。"""
    try:
        resp = call_generation_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            result_format="message",
        )
    except Exception as e:  # noqa: BLE001
        logger.error("DashScope Generation.call 最终失败: %s", e)
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        return

    for chunk in resp:
        if chunk.output is None:
            logger.error("DashScope 流式响应出错: %s", _response_error(chunk))
            yield f"data: {json.dumps({'error': _response_error(chunk)}, ensure_ascii=False)}\n\n"
            return
        delta = chunk.output.choices[0].message.content if chunk.output.choices else ""
        if delta:
            yield f"data: {json.dumps({'response': delta}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
