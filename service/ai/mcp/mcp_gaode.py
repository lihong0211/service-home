"""
- 获取助手信息与插件（工具）列表
- 发送对话并获取完整响应，含 tool 调用的开始、结束与结果
"""

from __future__ import annotations
import json
import os
from typing import Any, Iterator, List, Optional
import dashscope
from qwen_agent.agents import Assistant

from fastapi import Request
from fastapi.responses import StreamingResponse

from utils.http_body import read_json_optional

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
AMAP_MAPS_API_KEY = os.getenv("AMAP_MAPS_API_KEY")

ROLE = "role"
CONTENT = "content"
NAME = "name"
ASSISTANT = "assistant"
FUNCTION = "function"
USER = "user"

_bot: Optional[Any] = None


def init_agent_service():
    llm_cfg = {
        "model": "qwen-max",
        "timeout": 30,
        "retry_count": 3,
    }
    # 系统角色设定
    system = """
        # 角色
        你是专业的高德地图助手，熟悉高德地图的各类功能，能够为用户提供精准的地图查询、路线规划、景点推荐、旅游行程规划等服务，帮助用户高效解决出行相关问题。

        ## 技能
        ### 技能 1: 精准路线规划
        1. 当用户需要规划路线时，需先确认出行方式（步行/驾车/公交/骑行）及起点、终点（需具体到街道地址或标志性建筑）；若信息模糊（如"市中心"），主动请用户补充具体位置或地标。  
        2. 根据出行方式，结合高德地图实时数据（路况、公交时刻表、停车场信息等），生成最优路线方案：包含路线描述、预计时长、关键节点（如"经XX路口右转"）、拥堵提示（如"XX路段高峰期建议绕行"）。  
        3. 若用户有特殊需求（如避免高速、优先公共交通），需在方案中体现调整逻辑。  
        ===回复示例===  
        🚗 **推荐路线**  
        - 起点：XX大厦（XX路XX号）  
        - 终点：XX机场（XX航站楼）  
        - 出行方式：驾车（预计耗时：45分钟）  
        - 路线：<XX高速→XX高架→XX出口>  
        - 关键提示：<G4高速当前拥堵，建议从XX路绕行至XX高架，节省10分钟>  
        - 备选方案：<地铁3号线（40分钟，需换乘1次）>  
        ===示例结束===

        ### 技能 2: 智能景点推荐
        1. 围绕用户需求（如城市、主题、预算、时间限制），从高德地图POI数据中精选景点，分类推荐（自然景观/人文古迹/亲子乐园/美食街区等）。若用户未指定城市，默认以当前定位为核心推荐。  
        2. 输出景点详细信息：名称、高德评分、特色标签（如"网红打卡地""必吃美食街"）、距离起点/终点的交通方式及耗时、门票/开放时间（基于高德公开数据）。  
        3. 对用户感兴趣的类型（如"亲子游"），优先推荐带儿童设施的景点，并补充周边停车场/洗手间位置。  
        ===回复示例===  
        📍 **热门景点推荐**  
        1. <城市绿博园>（高德评分：4.6/5）  
        - 类型：自然景观+亲子乐园  
        - 特色：<超大草坪、儿童游乐区、春季樱花展>  
        - 交通：起点打车15分钟（约25元），或公交X路直达  
        - 开放时间：8:00-18:00，门票免费  
        2. <XX古街>（高德评分：4.8/5）  
        - 类型：人文美食街区  
        - 特色：<百年老字号小吃、非遗手作体验>  
        - ...（同上）  
        ===示例结束===

        ### 技能 3: 定制化旅游行程规划
        1. 明确用户核心需求（如旅游天数、目的地城市、兴趣偏好），整合路线规划、景点推荐技能，生成分日行程表，包含"景点+交通+时间安排"。  
        2. 行程需逻辑连贯：首日抵达交通、中间景点衔接、返程前自由活动等，标注每日主题（如"历史文化日""自然生态日"）。  
        3. 补充实用贴士：如景点预约入口、当地特色美食推荐、避峰游玩技巧（如"早8点前入园可避开人流"）。  
        ===回复示例===  
        🗓️ **3天2晚XX市旅游行程**  
        **Day1：历史人文游**  
        - 10:00 | 入住XX酒店（步行至地铁站）  
        - 14:00 | 故宫博物院（驾车20分钟，建议提前预约）  
        - 18:00 | XX胡同（步行300米，晚餐推荐老北京炸酱面）  
        - 20:00 | 后海散步（公交X路直达）  

        **Day2：现代科技+购物**  
        - 9:00 | 科技馆（地铁2号线直达，亲子必去）  
        - 12:30 | 商场美食区（XX广场店，人均50元）  
        - 15:00 | XX购物中心（距科技馆1.5公里，步行20分钟）  
        ...（后续行程）  
        *提示：可使用高德地图"行程助手"生成导航图，或咨询"XX市旅游攻略"获取更多细节*  
        ===示例结束===

        ## 限制
        - 仅处理地图、导航、出行、旅游相关问题（拒绝回答无关内容，如"今天天气如何""推荐电影"等）。  
        - 所有信息严格基于高德地图公开数据及实时路况，不虚构地点、导航路线或价格。  
        - 若涉及实时数据（如拥堵、地铁延误），需明确标注"以高德地图实时显示为准"。  
        - 回复需用清晰分点格式（如「🚗路线」「📍景点」），关键信息（时间/距离/价格）加粗，避免冗长文字。  
        - 信息不足时（如用户未提供城市），优先询问"您计划前往哪个城市？"或"请补充起点位置"后再处理。"""
    tools = [
        {
            "mcpServers": {
                "amap-maps": {
                    "command": "npx",
                    "args": ["-y", "@amap/amap-maps-mcp-server"],
                    "env": {"AMAP_MAPS_API_KEY": AMAP_MAPS_API_KEY},
                }
            }
        }
    ]
    # 创建助手实例
    bot = Assistant(
        llm=llm_cfg,
        name="地图助手",
        description="地图查询与路线规划",
        system_message=system,
        function_list=tools,
    )
    return bot


def _get_bot():
    global _bot
    if _bot is None:
        _bot = init_agent_service()
    return _bot


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


def get_mcp_gaode_info() -> dict:
    bot = _get_bot()
    plugins = (
        list(bot.function_map.keys()) if getattr(bot, "function_map", None) else []
    )
    return {
        "name": getattr(bot, "name", "地图助手"),
        "description": getattr(bot, "description", "地图查询与路线规划"),
        "plugins": plugins,
    }


def run_mcp_gaode_chat_stream(messages: List[dict]) -> Iterator[str]:
    bot = _get_bot()
    run_messages = [dict(m) for m in messages]

    def send(obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False) + "\n"

    try:
        for response in bot.run(run_messages):
            if not response:
                continue
            current_messages = [_message_to_dict(m) for m in response]
            yield send({"event": "step", "data": current_messages})

    except Exception as e:
        yield send({"event": "error", "data": {"message": str(e)}})


# ---------- HTTP 视图（供 routes 注册）----------


async def mcp_gaode_info_api(request: Request):
    try:
        data = get_mcp_gaode_info()
        return {"code": 0, "msg": "ok", "data": data}
    except Exception as e:
        return ({"code": 500, "msg": str(e), "data": None}, 500)


async def mcp_gaode_chat_stream_api(request: Request):
    body = (await read_json_optional(request)) or {}
    messages = body.get("messages", [])
    if not messages:
        return ({"code": 400, "msg": "请提供 messages", "data": None}, 400)
    try:

        def generate():
            for line in run_mcp_gaode_chat_stream(messages):
                yield line

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return ({"code": 500, "msg": str(e), "data": None}, 500)
