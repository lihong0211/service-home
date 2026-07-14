from fastapi import Request
from fastapi.responses import StreamingResponse

from service.ai._dashscope_common import stream_dashscope_sse


async def recipe_plan_api(request: Request):
    body = await request.json()
    ingredients = body.get("ingredients", "")
    servings = body.get("servings", 2)
    dietary = body.get("dietary", "普通")
    cuisine = body.get("cuisine", "家常菜")
    meal_type = body.get("meal_type", "午餐")

    if not ingredients:
        return {"code": 400, "msg": "请输入食材"}

    system_prompt = """你是一位专业营养师兼厨师，擅长根据现有食材设计美味健康的食谱。

食谱需包含：
1. 菜名和简介
2. 所需食材（含用量）
3. 详细烹饪步骤（分步说明）
4. 营养信息（热量/蛋白质/碳水/脂肪估算）
5. 烹饪小贴士
6. 如有剩余食材可做的其他菜品建议

使用 Markdown 格式，清晰易读。"""

    user_prompt = f"""我有以下食材：{ingredients}

请根据这些食材设计适合 {servings} 人份的 {meal_type}。
- 饮食类型：{dietary}
- 菜系偏好：{cuisine}

请给出 2-3 道菜的食谱方案。"""

    return StreamingResponse(
        stream_dashscope_sse(system_prompt, user_prompt), media_type="text/event-stream"
    )
