import json
from fastapi import Request
from fastapi.responses import StreamingResponse
from dashscope import Generation


async def startup_analyze_api(request: Request):
    body = await request.json()
    idea = body.get("idea", "")
    industry = body.get("industry", "")
    target_market = body.get("target_market", "")

    if not idea:
        return {"code": 400, "msg": "请输入创业想法"}

    system_prompt = """你是一位资深创业分析师，有丰富的投资和创业经验。
请对用户的创业想法进行全面分析，提供客观、深入的见解。

分析维度：
1. 市场机会（市场规模估算、增长趋势、时机判断）
2. 竞争格局（主要竞争对手、差异化空间）
3. 核心价值主张（用户痛点、解决方案的独特性）
4. 商业模式建议（收入模式、成本结构）
5. 主要风险与挑战（技术/市场/监管/资金）
6. 成功关键因素
7. 建议的下一步行动（验证假设的最小成本方式）

综合评分：市场吸引力/可行性/竞争壁垒（各10分制）

使用 Markdown，分析要有深度，不要泛泛而谈。"""

    user_prompt = f"""请分析以下创业想法：

**创业想法**：{idea}
{f'**所属行业**：{industry}' if industry else ''}
{f'**目标市场**：{target_market}' if target_market else ''}

请给出全面、专业的创业分析报告。"""

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
