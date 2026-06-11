import json
from fastapi import Request
from fastapi.responses import StreamingResponse
from dashscope import Generation


async def health_plan_api(request: Request):
    body = await request.json()
    age = body.get("age", 25)
    weight = body.get("weight", 65)
    height = body.get("height", 170)
    goal = body.get("goal", "保持健康")
    activity_level = body.get("activity_level", "轻度活动")
    health_issues = body.get("health_issues", "")

    bmi = weight / ((height / 100) ** 2)
    bmi_str = f"{bmi:.1f}"

    system_prompt = """你是一位专业的健康顾问，拥有营养学和运动科学双重背景。
你提供个性化的健康建议，但务必声明你不替代医疗诊断。

健身计划需包含：
1. 健康评估（BMI 分析、现状评估）
2. 每周训练计划（具体到每天的训练内容）
3. 饮食方案（三餐建议、营养搭配）
4. 水分和营养补充建议
5. 注意事项和安全提示
6. 4周进阶计划

使用 Markdown 格式。"""

    user_prompt = f"""请为我制定个性化健康计划：
- 年龄：{age} 岁
- 体重：{weight} kg，身高：{height} cm，BMI：{bmi_str}
- 活动水平：{activity_level}
- 健身目标：{goal}
{f'- 健康状况/限制：{health_issues}' if health_issues else ''}

请给出详细的健身和饮食计划。"""

    def generate():
        resp = Generation.call(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            result_format="message",
        )
        for chunk in resp:
            delta = chunk.output.choices[0].message.content if chunk.output.choices else ""
            if delta:
                yield f"data: {json.dumps({'response': delta}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
