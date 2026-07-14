from fastapi import Request
from fastapi.responses import StreamingResponse

from service.ai._dashscope_common import stream_dashscope_sse


async def travel_plan_api(request: Request):
    body = await request.json()
    destination = body.get("destination", "")
    days = body.get("days", 3)
    budget = body.get("budget", "适中")
    preferences = body.get("preferences", "")
    travel_style = body.get("travel_style", "文化体验")

    if not destination:
        return {"code": 400, "msg": "请输入目的地"}

    system_prompt = """你是一位经验丰富的旅行规划师，精通全球各地的旅游资源。
请根据用户需求，生成详细、实用的旅行攻略。

攻略需包含：
1. 目的地简介（2-3句）
2. 每日行程（按天列出，每天3-5个景点/活动）
3. 美食推荐（3-5道特色菜/餐厅）
4. 住宿建议（区域+价位参考）
5. 预算估算（交通/住宿/餐饮/景点门票）
6. 实用注意事项（签证/气候/交通/文化禁忌等）

格式要清晰，使用 Markdown，让读者一目了然。"""

    user_prompt = f"""请为我规划去 **{destination}** 的 {days} 天旅行攻略。
- 预算级别：{budget}
- 旅行风格：{travel_style}
{f'- 特殊偏好：{preferences}' if preferences else ''}

请生成完整的旅行攻略。"""

    return StreamingResponse(
        stream_dashscope_sse(system_prompt, user_prompt), media_type="text/event-stream"
    )
