import json
import random
from fastapi import Request
from fastapi.responses import StreamingResponse
from dashscope import Generation

MAJOR_ARCANA = [
    ("愚者 (The Fool)", "新开始、自由、冒险、天真"),
    ("魔术师 (The Magician)", "意志力、技能、创造力、力量"),
    ("女祭司 (The High Priestess)", "直觉、潜意识、神秘、内在智慧"),
    ("皇后 (The Empress)", "丰饶、母性、自然、创造"),
    ("皇帝 (The Emperor)", "权威、稳定、父性、领导"),
    ("教皇 (The Hierophant)", "传统、信仰、道德、指导"),
    ("恋人 (The Lovers)", "爱情、选择、联系、价值观"),
    ("战车 (The Chariot)", "胜利、控制、决心、前进"),
    ("力量 (Strength)", "内在力量、勇气、耐心、影响"),
    ("隐者 (The Hermit)", "独处、内省、引导、智慧"),
    ("命运之轮 (Wheel of Fortune)", "变化、循环、命运、转折"),
    ("正义 (Justice)", "公正、真理、因果、平衡"),
    ("吊人 (The Hanged Man)", "等待、牺牲、新视角、暂停"),
    ("死神 (Death)", "转变、结束与开始、蜕变"),
    ("节制 (Temperance)", "平衡、节制、耐心、调和"),
    ("恶魔 (The Devil)", "束缚、物质、阴影面、执念"),
    ("塔 (The Tower)", "突变、混乱、启示、解构"),
    ("星星 (The Star)", "希望、灵感、平静、更新"),
    ("月亮 (The Moon)", "幻觉、恐惧、无意识、不确定"),
    ("太阳 (The Sun)", "喜悦、成功、活力、清明"),
    ("审判 (Judgement)", "觉醒、反思、召唤、更新"),
    ("世界 (The World)", "完成、整合、成就、新阶段"),
]

MINOR_ARCANA_SUITS = ["权杖", "圣杯", "宝剑", "星币"]
MINOR_ARCANA = []
for suit in MINOR_ARCANA_SUITS:
    MINOR_ARCANA.append((f"{suit}王牌", f"{suit}的力量之源与纯粹潜能"))
    for num in ["2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        MINOR_ARCANA.append((f"{suit}{num}", f"{suit}的发展过程"))
    for face in ["侍者", "骑士", "王后", "国王"]:
        MINOR_ARCANA.append((f"{suit}{face}", f"{suit}人物的特质与能量"))

ALL_CARDS = MAJOR_ARCANA + MINOR_ARCANA

SPREAD_SIZES = {"single": 1, "three": 3, "celtic": 5}


async def tarot_read_api(request: Request):
    body = await request.json()
    question = body.get("question", "")
    spread_type = body.get("spread_type", "three")

    if not question:
        return {"code": 400, "msg": "请输入你的问题"}

    num_cards = SPREAD_SIZES.get(spread_type, 3)
    drawn = random.sample(ALL_CARDS, num_cards)
    is_reversed = [random.random() < 0.3 for _ in drawn]

    cards_info = []
    for i, (name, meaning) in enumerate(drawn):
        rev = is_reversed[i]
        cards_info.append({
            "name": name,
            "meaning": meaning,
            "reversed": rev,
            "position": ["过去", "现在", "未来", "建议", "结果"][i] if num_cards > 1 else "答案",
        })

    cards_desc = "\n".join(
        f"- {c['position']}：{c['name']}（{'逆位' if c['reversed'] else '正位'}）- {c['meaning']}"
        for c in cards_info
    )

    system_prompt = """你是一位深谙塔罗之道的占卜师，拥有丰富的神秘学知识。
你的解读充满诗意和智慧，既尊重传统意义，又结合现代心理学洞察。
请用温暖而深邃的语气解读牌意，让提问者获得真正有价值的启示。"""

    user_prompt = f"""提问者的问题：{question}

抽到的牌：
{cards_desc}

请进行深入解读，结合每张牌的位置和正逆位，给出有洞察力的分析和建议。"""

    cards_response = cards_info

    def generate():
        yield f"data: {json.dumps({'type': 'cards', 'cards': cards_response}, ensure_ascii=False)}\n\n"

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
                yield f"data: {json.dumps({'type': 'reading', 'response': delta}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
