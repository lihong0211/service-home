import json
from fastapi import Request
from fastapi.responses import StreamingResponse

from service.ai._dashscope_common import call_generation_with_retry

SCENARIOS = {
    "car": {
        "name": "买车谈判",
        "context": "二手车市场，一辆2020年大众帕萨特，车主报价12万，车况良好但有轻微划痕",
        "buyer_hint": "你是买家，目标价10万以内，可接受11万",
        "seller_hint": "你是卖家，底价10.5万，报价12万，需要在本周内出售",
    },
    "salary": {
        "name": "薪资谈判",
        "context": "候选人参加某互联网公司后端工程师面试，HR给出offer：月薪25K",
        "buyer_hint": "你是候选人，目前薪资20K，期望30K，可接受27K",
        "seller_hint": "你是HR，薪资范围20-28K，倾向于25K成交",
    },
    "rent": {
        "name": "租房谈判",
        "context": "上海浦东某两居室，房东报价6000元/月，配套齐全",
        "buyer_hint": "你是租客，预算5000元，可接受5500元，希望免押金",
        "seller_hint": "你是房东，底价5200元，希望一次性付半年",
    },
    "contract": {
        "name": "商业合同",
        "context": "软件开发外包合同，乙方提供定制化ERP系统开发服务，报价50万",
        "buyer_hint": "你是甲方，预算35万，可接受42万，重视交付周期",
        "seller_hint": "你是乙方，成本30万，报价50万，底价40万",
    },
}


async def negotiation_chat_api(request: Request):
    body = await request.json()
    message = body.get("message", "")
    scenario_key = body.get("scenario", "car")
    user_role = body.get("user_role", "buyer")
    history = body.get("history", [])

    if not message:
        return {"code": 400, "msg": "请输入内容"}

    scenario = SCENARIOS.get(scenario_key, SCENARIOS["car"])
    ai_role = "seller" if user_role == "buyer" else "buyer"
    ai_role_name = "卖家" if ai_role == "seller" else "买家"
    user_role_name = "买家" if user_role == "buyer" else "卖家"
    ai_hint = scenario["seller_hint"] if ai_role == "seller" else scenario["buyer_hint"]

    system_prompt = f"""你正在参与一场真实的商业谈判模拟。

场景：{scenario['name']}
背景：{scenario['context']}

你扮演：{ai_role_name}
{ai_hint}

谈判规则：
- 保持角色，不要出戏
- 展示真实的谈判策略（锚定/让步/拒绝/反报价等）
- 回复简洁，像真实对话（2-4句话）
- 可以适当强硬，也可以适当妥协，但要有策略
- 不要主动放弃底线，等对方施压
- 记住你的底价，不要低于底线成交"""

    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-10:]:
        messages.append({"role": "user" if h["role"] == user_role_name else "assistant", "content": h["content"]})
    messages.append({"role": "user", "content": message})

    def generate():
        resp = call_generation_with_retry(
            model="qwen-turbo",
            messages=messages,
            stream=True,
            result_format="message",
        )
        for chunk in resp:
            if chunk.output is None:
                yield f"data: {json.dumps({'error': str(chunk.message)}, ensure_ascii=False)}\n\n"
                return
            delta = chunk.output.choices[0].message.content if chunk.output.choices else ""
            if delta:
                yield f"data: {json.dumps({'response': delta, 'ai_role': ai_role_name}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def list_scenarios_api(request: Request):
    return {
        "code": 0,
        "msg": "success",
        "data": [
            {"key": k, "name": v["name"], "context": v["context"]}
            for k, v in SCENARIOS.items()
        ],
    }
