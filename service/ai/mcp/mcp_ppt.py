"""
MCP PPT 助手：通过 MCP 接入 ChatPPT 官方服务（YOOTeam/ChatPPT-MCP）
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Iterator, List, Optional, Tuple

import dashscope
from qwen_agent.agents import Assistant

_SSL_RETRY_COUNT = 3
_SSL_RETRY_DELAY = 1.5  # seconds, doubles each attempt


def _is_ssl_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "ssl" in msg or "eof occurred" in msg or "unexpected_eof" in msg


def _collect_bot_run(bot: Any, messages: List[dict]) -> Tuple[List, Optional[Exception]]:
    """Run bot.run() with automatic retry on transient SSL errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(_SSL_RETRY_COUNT):
        try:
            steps = []
            for response in bot.run(messages):
                steps.append(response)
            return steps, None
        except Exception as e:
            if _is_ssl_error(e) and attempt < _SSL_RETRY_COUNT - 1:
                last_exc = e
                time.sleep(_SSL_RETRY_DELAY * (attempt + 1))
                continue
            return [], e
    return [], last_exc

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

DEFAULT_CHATPPT_MCP_URL = "http://mcp.yoo-ai.com/mcp"
YOO_API_KEY = os.getenv("YOO_API_KEY")
YOO_API_BASE = "https://saas.api.yoo-ai.com"

# PPT 生成状态码
PPT_STATUS_PENDING = 0
PPT_STATUS_PROCESSING = 1
PPT_STATUS_SUCCESS = 2
PPT_STATUS_FAILED = 3


def _yoo_headers() -> dict:
    return {"Authorization": f"Bearer {YOO_API_KEY}"}


def query_ppt_status(ppt_id: str) -> dict:
    """直接查询 PPT 生成进度（无需经过 LLM）。
    status: 1=生成中, 2=成功, 3=失败
    """
    import requests
    resp = requests.get(
        f"{YOO_API_BASE}/mcp/ppt/ppt-result",
        params={"id": ppt_id},
        headers=_yoo_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_ppt_download_url(ppt_id: str) -> dict:
    """获取 PPT 下载地址（仅 status=2 时有效）。"""
    import requests
    resp = requests.get(
        f"{YOO_API_BASE}/mcp/ppt/ppt-download",
        params={"id": ppt_id},
        headers=_yoo_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def get_ppt_editor_url(ppt_id: str) -> dict:
    """获取 PPT 在线编辑器地址。"""
    import requests
    resp = requests.post(
        f"{YOO_API_BASE}/mcp/ppt/ppt-editor",
        data={"id": ppt_id},
        headers=_yoo_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


ROLE = "role"
CONTENT = "content"
NAME = "name"
ASSISTANT = "assistant"
FUNCTION = "function"
USER = "user"

_bot: Optional[Any] = None


def _build_chatppt_mcp_config() -> Optional[dict]:
    url = f"{DEFAULT_CHATPPT_MCP_URL}?key={YOO_API_KEY}"

    return {
        "mcpServers": {
            "chatppt": {
                "type": "streamable-http",
                "url": url,
                "sse_read_timeout": 300,
            }
        }
    }


def init_agent_service():
    mcp_config = _build_chatppt_mcp_config()
    llm_cfg = {
        "model": "qwen-max",
        "timeout": 60,
        "retry_count": 3,
    }
    system = (
        "你是 PPT 汇报助手，已接入 ChatPPT MCP 服务。"
        "你可以根据用户主题或需求生成 PPT，也支持上传文档自动生成演示文稿、在线编辑与下载。"
        "当用户说「基于本月销售数据做个汇报PPT」或类似需求时，使用 MCP 提供的工具生成或优化 PPT，并告知用户如何查看或下载。\n\n"
        "【回复格式要求】\n"
        "- 只使用纯文本或 Markdown 格式回复，严禁输出任何 HTML 标签（如 <iframe>、<details>、<summary>、<div> 等）。\n"
        "- 严禁在回复中嵌入 SVG、base64 图片或任何预览代码。\n"
        "- PPT 生成任务完成后，只需告知用户：ppt_id、页数、标题，以及「可点击页面上的预览/下载按钮查看」，不要自行生成预览内容。\n"
        "- 如果 PPT 仍在生成中（status=1），告知用户正在生成，请稍候，不要输出进度 HTML。"
    )
    tools = [mcp_config]
    bot = Assistant(
        llm=llm_cfg,
        name="PPT 汇报助手",
        description="接入 ChatPPT MCP，根据主题或文档生成、编辑与下载 PPT。可接入活字格、阿里云百炼等。",
        system_message=system,
        function_list=tools,
    )
    return bot


def _get_bot() -> Any:
    global _bot
    if _bot is None:
        _bot = init_agent_service()
    return _bot


def reset_bot() -> None:
    """强制重建 bot 实例（系统提示词变更后调用）。"""
    global _bot
    _bot = None


def _message_to_dict(msg: Any) -> dict:
    if isinstance(msg, dict):
        role = msg.get(ROLE, "")
        content = msg.get(CONTENT, "")
        name = msg.get(NAME)
        fn_call = msg.get("function_call")
    else:
        role = getattr(msg, ROLE, "")
        content = getattr(msg, CONTENT, "")
        name = getattr(msg, NAME, None)
        fn_call = getattr(msg, "function_call", None)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                text_parts.append(item.text or "")
            else:
                text_parts.append(str(item))
        content = " ".join(text_parts)
    if content is None:
        content = ""
    out = {ROLE: role, CONTENT: content}
    if name:
        out[NAME] = name
    if fn_call:
        if hasattr(fn_call, "name"):
            out["function_call"] = {
                "name": fn_call.name,
                "arguments": getattr(fn_call, "arguments", "{}"),
            }
        else:
            out["function_call"] = (
                fn_call
                if isinstance(fn_call, dict)
                else {"name": "", "arguments": "{}"}
            )
    return out


def get_mcp_ppt_info() -> dict:
    """返回 PPT 助手元信息与 MCP 插件列表。未配置或连接失败时返回配置说明。"""
    try:
        bot = _get_bot()
        plugins = (
            list(bot.function_map.keys()) if getattr(bot, "function_map", None) else []
        )
        return {
            "name": getattr(bot, "name", "PPT 汇报助手"),
            "description": getattr(
                bot,
                "description",
                "接入 ChatPPT MCP，根据主题或文档生成、编辑与下载 PPT。",
            ),
            "plugins": plugins,
            "mcp_server": "chatppt",
        }
    except ValueError as e:
        return {
            "name": "PPT 汇报助手",
            "description": "需配置 ChatPPT MCP 后使用。",
            "plugins": [],
            "mcp_server": None,
            "config_required": True,
            "config_hint": str(e),
        }
    except Exception as e:
        return {
            "name": "PPT 汇报助手",
            "description": "需配置 ChatPPT MCP 后使用。",
            "plugins": [],
            "mcp_server": None,
            "config_required": True,
            "config_hint": str(e),
        }


def run_mcp_ppt_chat_stream(messages: List[dict]) -> Iterator[str]:
    """流式返回与 ChatPPT MCP 的对话步骤（NDJSON）。
    每个 step 实时 yield，保持真正的流式体验。
    SSL 抖动在此层 catch 后 yield error event，不阻塞已发送内容。
    """

    def send(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    try:
        bot = _get_bot()
    except ValueError as e:
        yield send({"event": "error", "data": {"message": str(e)}})
        return

    run_messages = [dict(m) for m in messages]
    try:
        for response in bot.run(run_messages):
            if not response:
                continue
            current_messages = [_message_to_dict(m) for m in response]
            yield send({"event": "step", "data": current_messages})
    except Exception as e:
        err_msg = "网络抖动，请重新发送" if _is_ssl_error(e) else str(e)
        yield send({"event": "error", "data": {"message": err_msg}})


def run_mcp_ppt_chat(
    messages: List[dict],
    model: str = "qwen-turbo",
    system_message: Optional[str] = None,
) -> dict:
    """
    执行与 ChatPPT MCP 的对话，收集全部步骤并返回。
    model / system_message 当前由 Assistant 初始化固定，保留参数以兼容路由。
    """
    try:
        bot = _get_bot()
    except ValueError as e:
        return {
            "error": str(e),
            "reply_messages": [],
            "steps": [],
            "history": [],
            "final_answer": "",
        }

    run_messages = [dict(m) for m in messages]
    steps = []
    final_answer = ""
    history = []
    all_responses, exc = _collect_bot_run(bot, run_messages)
    if exc is not None:
        return {
            "error": str(exc),
            "reply_messages": [],
            "steps": steps,
            "history": history,
            "final_answer": final_answer,
        }
    for response in all_responses:
        if not response:
            continue
        current = [_message_to_dict(m) for m in response]
        steps.append({"type": "step", "data": current})
        history = current
    for msg in reversed(history or []):
        if msg.get(ROLE) == ASSISTANT and msg.get(CONTENT):
            final_answer = (msg.get(CONTENT) or "").strip()
            break

    reply_messages = [m for m in history if m.get(ROLE) in (ASSISTANT, USER)]
    result = {
        "reply_messages": reply_messages,
        "steps": steps,
        "history": history,
        "final_answer": final_answer,
    }

    # 异步写入生成记录（提取 ppt_id，失败不影响主流程）
    _try_save_ppt_record(
        steps=steps,
        final_answer=final_answer,
        prompt=messages[-1].get(CONTENT, "") if messages else "",
    )

    return result


def _extract_ppt_id(obj) -> Optional[str]:
    """从任意嵌套 dict/str 中递归提取 ppt_id。
    兼容结构：
      {"ppt_id": "xxx"}
      {"id": "xxx"}
      {"data": {"ppt_id": "xxx"}}
      {"data": {"id": "xxx"}}
      {"code":200, "data": {"id": "xxx"}}
    """
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None

    # 直接字段
    for key in ("ppt_id", "id"):
        val = obj.get(key)
        if val and isinstance(val, str) and len(val) > 8:
            return val

    # 嵌套一层 data
    data = obj.get("data")
    if isinstance(data, dict):
        for key in ("ppt_id", "id"):
            val = data.get(key)
            if val and isinstance(val, str) and len(val) > 8:
                return val

    return None


def _extract_title_from_steps(steps: list, final_answer: str) -> Optional[str]:
    """从 function 结果或 final_answer 中提取 ppt 标题。"""
    # 1) function 结果里的 ppt_title / title
    for step in steps:
        for msg in step.get("data", []):
            if msg.get(ROLE) != FUNCTION:
                continue
            try:
                parsed = json.loads(msg.get(CONTENT) or "{}")
                if isinstance(parsed, dict):
                    for key in ("ppt_title", "title"):
                        t = parsed.get(key)
                        if t and isinstance(t, str) and len(t) < 200:
                            return t.strip()
                    data = parsed.get("data")
                    if isinstance(data, dict):
                        for key in ("ppt_title", "title"):
                            t = data.get(key)
                            if t and isinstance(t, str) and len(t) < 200:
                                return t.strip()
            except Exception:
                pass
    # 2) final_answer 里 "标题: xxx" 或 "标题：xxx"
    if final_answer:
        m = re.search(r"标题[：:]\s*([^\n\r]+)", final_answer)
        if m:
            return m.group(1).strip()[:200]
    return None


def _try_save_ppt_record(steps: list, final_answer: str, prompt: str) -> None:
    """从 steps / final_answer 中提取 ppt_id、标题 并写入 ppt_record 表。"""
    try:
        ppt_id = None

        # 优先从 function 调用结果里取（兼容所有嵌套结构）
        for step in steps:
            for msg in step.get("data", []):
                if msg.get(ROLE) == FUNCTION:
                    ppt_id = _extract_ppt_id(msg.get(CONTENT) or "{}")
                    if ppt_id:
                        break
            if ppt_id:
                break

        # 兜底1：正则从 final_answer 里匹配 ppt_id / 任务ID
        if not ppt_id and final_answer:
            for pattern in (
                r'"ppt_id"\s*:\s*"([A-Za-z0-9_\-]{8,})"',
                r'"id"\s*:\s*"([A-Za-z0-9_\-]{8,})"',
                r'任务ID[：:]\s*([A-Za-z0-9_\-]{8,})',
                r'ppt_id[：: ]+([A-Za-z0-9_\-]{8,})',
            ):
                m = re.search(pattern, final_answer)
                if m:
                    ppt_id = m.group(1)
                    break

        if not ppt_id:
            print(f"[PptRecord] ppt_id 未找到，steps={len(steps)}，final_answer前100={final_answer[:100]}", flush=True)
            return

        from model.ai.ppt_record import PptRecord
        from app.app import db

        existing = PptRecord.select_one_by({"ppt_id": ppt_id})
        if existing:
            print(f"[PptRecord] 已存在，跳过 ppt_id={ppt_id}", flush=True)
            return

        title = _extract_title_from_steps(steps, final_answer)
        if not title and prompt:
            title = prompt.strip()[:80]  # 用用户输入当临时标题

        record = PptRecord()
        record.ppt_id = ppt_id
        record.title = title or ""
        record.prompt = prompt[:500] if prompt else ""
        record.status = 1
        db.session.add(record)
        db.session.commit()
        print(f"[PptRecord] 写入成功 ppt_id={ppt_id} title={title or '(无)'}", flush=True)
    except Exception as e:
        print(f"[PptRecord] 写入失败: {e}", flush=True)


# ---------- Flask 视图（供 routes 直接注册）----------

def mcp_ppt_info_api():
    from flask import jsonify
    try:
        data = get_mcp_ppt_info()
        return jsonify({"code": 0, "msg": "ok", "data": data})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500


def mcp_ppt_chat_api():
    from flask import request, jsonify
    body = request.get_json() or {}
    messages = body.get("messages", [])
    if not messages:
        return jsonify({"code": 400, "msg": "请提供 messages", "data": None}), 400
    try:
        out = run_mcp_ppt_chat(
            messages,
            model=body.get("model", "qwen-turbo"),
            system_message=body.get("system_message"),
        )
        if out.get("error"):
            return jsonify({"code": 500, "msg": out["error"], "data": out}), 500
        return jsonify({"code": 0, "msg": "ok", "data": out})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500


def mcp_ppt_chat_stream_api():
    from flask import request, jsonify, Response, stream_with_context
    body = request.get_json() or {}
    messages = body.get("messages", [])
    if not messages:
        return jsonify({"code": 400, "msg": "请提供 messages", "data": None}), 400
    try:
        def generate():
            for line in run_mcp_ppt_chat_stream(messages):
                yield line
        return Response(
            stream_with_context(generate()),
            mimetype="application/x-ndjson",
            headers={"X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500


def mcp_ppt_status_api():
    """GET /ai/mcp-ppt/status?ppt_id=xxx
    直接查询 PPT 生成进度，status: 1=生成中, 2=成功, 3=失败。
    """
    from flask import request, jsonify
    ppt_id = request.args.get("ppt_id") or (request.get_json(silent=True) or {}).get("ppt_id")
    if not ppt_id:
        return jsonify({"code": 400, "msg": "请提供 ppt_id"}), 400
    try:
        data = query_ppt_status(ppt_id)
        # 同步更新本地记录（状态/标题/页数/preview_url）
        _sync_ppt_record(ppt_id, data)
        return jsonify({"code": 0, "msg": "ok", "data": data})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


def _sync_ppt_record(ppt_id: str, api_data: dict) -> None:
    """将 YOO API 返回的最新状态同步到本地 ppt_record 表。"""
    try:
        from model.ai.ppt_record import PptRecord
        from app.app import db

        raw = api_data.get("data") or api_data
        if not isinstance(raw, dict):
            return

        record = PptRecord.select_one_by({"ppt_id": ppt_id})
        if not record:
            record = PptRecord()
            record.ppt_id = ppt_id
            db.session.add(record)

        record.status      = raw.get("status", record.status or 1)
        record.page_count  = raw.get("page_count") or record.page_count
        record.preview_url = raw.get("preview_url") or record.preview_url
        record.process_url = raw.get("process_url") or record.process_url
        if raw.get("ppt_title"):
            record.title = raw["ppt_title"]
        db.session.commit()
    except Exception:
        pass


def mcp_ppt_download_url_api():
    """GET /ai/mcp-ppt/download-url?ppt_id=xxx
    获取 PPT 下载地址（需 status=2 完成后调用）。
    """
    from flask import request, jsonify
    ppt_id = request.args.get("ppt_id") or (request.get_json(silent=True) or {}).get("ppt_id")
    if not ppt_id:
        return jsonify({"code": 400, "msg": "请提供 ppt_id"}), 400
    try:
        data = get_ppt_download_url(ppt_id)
        return jsonify({"code": 0, "msg": "ok", "data": data})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


def mcp_ppt_download_proxy_api():
    """GET /ai/mcp-ppt/download?ppt_id=xxx&out_trade_no=xxx
    通过后端代理将 PPT 文件流式传输给客户端，客户端直接触发浏览器下载。
    需携带已支付的 out_trade_no，否则返回支付引导。
    """
    import requests as req
    from flask import request, jsonify, Response, stream_with_context
    from model.payment.pay_order import PayOrder

    ppt_id = request.args.get("ppt_id")
    out_trade_no = request.args.get("out_trade_no")

    if not ppt_id:
        return jsonify({"code": 400, "msg": "请提供 ppt_id"}), 400

    # 支付核验
    if not out_trade_no:
        return jsonify({"code": 402, "msg": "请先完成支付，传入 out_trade_no"}), 402

    order = PayOrder.select_one_by({"out_trade_no": out_trade_no, "biz_id": ppt_id})
    if not order:
        return jsonify({"code": 404, "msg": "订单不存在"}), 404
    if order.status != 2:
        status_hint = {0: "请先完成微信扫码付款", 1: "付款审核中，请稍候（通常几分钟内）", 3: "订单已关闭"}
        return jsonify({"code": 402, "msg": status_hint.get(order.status, "订单未确认")}), 402

    try:
        url_resp = get_ppt_download_url(ppt_id)
        raw = url_resp.get("data") or url_resp
        # 官方文档字段为 download_url，兼容 url / ppt_url
        download_url = (
            raw.get("download_url")
            or raw.get("url")
            or raw.get("ppt_url")
        ) if isinstance(raw, dict) else None
        if not download_url:
            return jsonify({"code": 400, "msg": "PPT 尚未生成完成或无下载地址", "raw": url_resp}), 400
        r = req.get(download_url, stream=True, timeout=120)
        r.raise_for_status()
        filename = f"{ppt_id}.pptx"
        return Response(
            stream_with_context(r.iter_content(chunk_size=8192)),
            content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


def mcp_ppt_editor_url_api():
    """GET /ai/mcp-ppt/editor?ppt_id=xxx
    获取 PPT 在线编辑器地址。
    """
    from flask import request, jsonify
    ppt_id = request.args.get("ppt_id") or (request.get_json(silent=True) or {}).get("ppt_id")
    if not ppt_id:
        return jsonify({"code": 400, "msg": "请提供 ppt_id"}), 400
    try:
        data = get_ppt_editor_url(ppt_id)
        return jsonify({"code": 0, "msg": "ok", "data": data})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


def mcp_ppt_history_api():
    """GET /ai/mcp-ppt/history?page=1&page_size=20
    查询本地 PPT 生成历史记录列表（按创建时间倒序）。
    """
    from flask import request, jsonify
    from model.ai.ppt_record import PptRecord
    from app.app import db

    page      = max(int(request.args.get("page", 1)), 1)
    page_size = min(int(request.args.get("page_size", 20)), 100)
    offset    = (page - 1) * page_size

    try:
        total = PptRecord.count()
        records = (
            db.session.query(PptRecord)
            .filter(PptRecord.deleted_at.is_(None))
            .order_by(PptRecord.create_at.desc())
            .offset(offset)
            .limit(page_size)
            .all()
        )
        def _display_title(r) -> str:
            """列表展示用：优先标题，否则用 prompt 前 30 字，否则「PPT生成中」"""
            if r.title and r.title.strip():
                return r.title.strip()
            if r.prompt and r.prompt.strip():
                return (r.prompt.strip()[:30] + "…") if len(r.prompt.strip()) > 30 else r.prompt.strip()
            return "PPT生成中"

        return jsonify({
            "code": 0,
            "msg": "ok",
            "data": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "list": [
                    {
                        "ppt_id":       r.ppt_id,
                        "title":        r.title or "",
                        "display_title": _display_title(r),  # 前端列表直接用此字段展示，避免显示 ppt_id
                        "prompt":       r.prompt or "",
                        "page_count":   r.page_count,
                        "status":       r.status,
                        "preview_url":  r.preview_url or "",
                        "create_at":    r.create_at.strftime("%Y-%m-%d %H:%M:%S") if r.create_at else "",
                    }
                    for r in records
                ],
            },
        })
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500
