from fastapi import Request
from fastapi.responses import StreamingResponse

from service.ai._dashscope_common import stream_dashscope_sse


async def wellbeing_chat_api(request: Request):
    body = await request.json()
    message = body.get("message", "")
    session_id = body.get("session_id", "")
    mood = body.get("mood", "")

    if not message:
        return {"code": 400, "msg": "请输入内容"}

    mood_context = f"用户当前心情状态：{mood}。" if mood else ""

    system_prompt = f"""你是一位温暖、专业的心理健康助手，受过正规心理咨询培训。
{mood_context}

你的工作方式：
- 首先倾听和理解，表达真诚的共情
- 用温和、非评判的语气回应
- 帮助用户识别和理解自己的情绪
- 提供实用的情绪调节技巧（如正念、深呼吸、认知重构）
- 必要时鼓励寻求专业心理咨询或医疗帮助

重要声明：你的支持不替代专业的心理治疗或医疗诊断。
如果用户表达有自伤或伤害他人的想法，必须立即建议拨打心理援助热线（如：北京 010-82951332，全国 400-161-9995）。"""

    return StreamingResponse(
        stream_dashscope_sse(system_prompt, message), media_type="text/event-stream"
    )
