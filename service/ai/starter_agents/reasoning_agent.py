import json
import re
from fastapi import Request
from fastapi.responses import StreamingResponse
import httpx


async def reasoning_chat_api(request: Request):
    body = await request.json()
    question = body.get("question", "")
    model = body.get("model", "deepseek-r1:1.5b")

    if not question:
        return {"code": 400, "msg": "请输入问题"}

    async def generate():
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": question, "stream": True},
            ) as resp:
                think_buf = ""
                answer_buf = ""
                in_think = False

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue

                    token = data.get("response", "")
                    if not token:
                        continue

                    think_buf += token
                    answer_buf += token

                    # detect <think> open
                    if "<think>" in think_buf and not in_think:
                        in_think = True
                        think_buf = think_buf[think_buf.index("<think>") + 7:]
                        answer_buf = ""

                    if in_think:
                        if "</think>" in think_buf:
                            close_idx = think_buf.index("</think>")
                            chunk_content = think_buf[:close_idx]
                            if chunk_content:
                                yield f"data: {json.dumps({'type': 'think', 'content': chunk_content}, ensure_ascii=False)}\n\n"
                            in_think = False
                            remainder = think_buf[close_idx + 8:]
                            think_buf = ""
                            answer_buf = remainder
                            if remainder:
                                yield f"data: {json.dumps({'type': 'answer', 'content': remainder}, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'think', 'content': token}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'answer', 'content': token}, ensure_ascii=False)}\n\n"

                    if data.get("done"):
                        break

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
