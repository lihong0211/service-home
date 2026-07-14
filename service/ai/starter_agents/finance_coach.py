from fastapi import Request
from fastapi.responses import StreamingResponse

from service.ai._dashscope_common import stream_dashscope_sse


async def finance_plan_api(request: Request):
    body = await request.json()
    monthly_income = body.get("monthly_income", 0)
    monthly_expenses = body.get("monthly_expenses", 0)
    savings_goal = body.get("savings_goal", 0)
    debt = body.get("debt", 0)
    investment_risk = body.get("investment_risk", "稳健型")
    financial_goals = body.get("financial_goals", "")

    monthly_savings = monthly_income - monthly_expenses
    savings_rate = (monthly_savings / monthly_income * 100) if monthly_income > 0 else 0

    system_prompt = """你是一位持牌的专业理财顾问，擅长个人财务规划。
请提供实际可操作的财务建议，但注意：投资有风险，建议仅供参考，不构成投资决策依据。

财务规划需包含：
1. 财务健康评估（收支分析、储蓄率评价）
2. 支出优化建议（哪些可以削减）
3. 紧急备用金建议
4. 储蓄和投资策略（根据风险偏好）
5. 债务管理方案（如有）
6. 3年/5年/10年财务里程碑
7. 具体行动清单

使用 Markdown，数字清晰，建议具体可执行。"""

    user_prompt = f"""请为我制定个人财务规划：
- 月收入：¥{monthly_income:,.0f}
- 月支出：¥{monthly_expenses:,.0f}
- 每月可储蓄：¥{monthly_savings:,.0f}（储蓄率：{savings_rate:.1f}%）
- 储蓄目标：¥{savings_goal:,.0f}
- 当前负债：¥{debt:,.0f}
- 投资风险偏好：{investment_risk}
- 财务目标：{financial_goals}

请给出详细的财务规划方案。"""

    return StreamingResponse(
        stream_dashscope_sse(system_prompt, user_prompt), media_type="text/event-stream"
    )
