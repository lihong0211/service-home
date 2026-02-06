"""基于 Assistant 实现的高德地图智能助手

这个模块提供了一个智能地图助手，可以：
1. 通过自然语言进行地图服务查询
2. 支持多种交互方式（GUI、TUI、测试模式）
3. 支持旅游规划、地点查询、路线导航等功能
"""

import os
import json
import time
from typing import Optional, Generator
from qwen_agent.gui import WebUI

# 处理代理问题：在导入 dashscope 之前配置代理设置
# 如果设置了 DASHSCOPE_DISABLE_PROXY，则完全禁用代理
if os.getenv("DASHSCOPE_DISABLE_PROXY", "").lower() in ("true", "1", "yes"):
    # 清除所有代理相关环境变量
    for proxy_var in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]:
        if proxy_var in os.environ:
            del os.environ[proxy_var]
    # 设置 NO_PROXY 为所有域名，禁用代理
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
elif "NO_PROXY" not in os.environ and "no_proxy" not in os.environ:
    # 如果没有明确设置，默认禁用代理以避免连接问题
    # 如果需要使用代理，请通过环境变量明确配置
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"

# 现在导入 dashscope（代理设置已配置）
import dashscope
from qwen_agent.agents import Assistant
from flask import Response, stream_with_context

# 配置 dashscope API key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
AMAP_MAPS_API_KEY = os.getenv("AMAP_MAPS_API_KEY")

# 全局 bot 实例（单例模式，避免重复初始化）
_bot_instance = None


def init_agent_service():
    """初始化高德地图助手服务

    配置说明：
    - 使用 qwen-max 作为底层语言模型
    - 设置系统角色为地图助手
    - 配置高德地图 MCP 工具

    Returns:
        Assistant: 配置好的地图助手实例
    """
    # LLM 模型配置
    llm_cfg = {
        "model": "qwen-max",
        "timeout": 30,  # 设置模型调用超时时间
        # "retry_count": 3,  # 设置重试次数
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
    # MCP 工具配置
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
    print("助手初始化成功！")
    return bot


def get_bot_instance():
    """获取 bot 实例（单例模式）"""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = init_agent_service()
    return _bot_instance


def get_available_tools():
    """获取可用的MCP工具列表

    Returns:
        dict: 包含工具列表的字典
    """
    try:
        bot = get_bot_instance()
        tools_list = []

        # 从bot实例中获取工具信息
        # qwen-agent的Assistant可能通过function_list或内部属性存储工具信息
        if hasattr(bot, "function_list") and bot.function_list:
            for tool_config in bot.function_list:
                if isinstance(tool_config, dict) and "mcpServers" in tool_config:
                    for server_name, server_config in tool_config["mcpServers"].items():
                        # 这里列出已知的高德地图MCP工具
                        # 实际工具列表需要从MCP服务器动态获取
                        amap_tools = [
                            "amap-maps-maps_regeocode",
                            "amap-maps-maps_geo",
                            "amap-maps-maps_ip_location",
                            "amap-maps-maps_weather",
                            "amap-maps-maps_search_detail",
                            "amap-maps-maps_bicycling",
                            "amap-maps-maps_direction_walking",
                            "amap-maps-maps_direction_driving",
                            "amap-maps-maps_direction_transit_integrated",
                            "amap-maps-maps_distance",
                            "amap-maps-maps_text_search",
                            "amap-maps-maps_around_search",
                        ]
                        tools_list.extend(amap_tools)

        return {
            "code": 200,
            "data": {
                "assistant_name": bot.name if hasattr(bot, "name") else "地图助手",
                "assistant_description": (
                    bot.description
                    if hasattr(bot, "description")
                    else "地图查询与路线规划"
                ),
                "tools": sorted(list(set(tools_list))),  # 去重并排序
                "mcp_servers": ["amap-maps"],
            },
        }
    except Exception as e:
        return {"code": 500, "msg": f"获取工具列表失败: {str(e)}", "data": None}


def chat_stream(
    query: str, file_url: Optional[str] = None, messages: Optional[list] = None
) -> Generator[str, None, None]:
    """流式聊天接口

    根据 bot.run() 返回的流式结果，设计优化的流式返回接口。
    每次返回增量内容（delta）和完整内容，以及 finish_reason 等信息。

    Args:
        query: 用户问题
        file_url: 可选的文件URL
        messages: 可选的对话历史

    Yields:
        str: Server-Sent Events 格式的流式响应数据
    """
    try:
        bot = get_bot_instance()

        # 构建消息
        if messages is None:
            messages = []

        if not file_url:
            messages.append({"role": "user", "content": query})
        else:
            messages.append(
                {"role": "user", "content": [{"text": query}, {"file": file_url}]}
            )

        # 用于跟踪上一次的内容，计算增量
        last_content = ""
        last_full_response = None
        request_id = None
        created_time = int(time.time())

        # 用于跟踪工具调用
        tool_calls = []
        used_tools = set()

        # 流式返回响应
        for response in bot.run(messages):
            last_full_response = response

            # 处理响应（可能是列表或单个对象）
            items = response if isinstance(response, list) else [response]

            for item in items:
                current_content = item.get("content", "")
                role = item.get("role", "assistant")
                name = item.get("name", "地图助手")
                extra = item.get("extra", {})

                # 检查是否有工具调用信息
                # qwen-agent可能在extra中存储工具调用信息
                if "tool_calls" in item or "function_calls" in item:
                    tool_call_info = item.get("tool_calls") or item.get(
                        "function_calls"
                    )
                    if tool_call_info:
                        for tool_call in (
                            tool_call_info
                            if isinstance(tool_call_info, list)
                            else [tool_call_info]
                        ):
                            tool_name = tool_call.get("name") or tool_call.get(
                                "function", {}
                            ).get("name", "")
                            tool_input = tool_call.get("arguments") or tool_call.get(
                                "function", {}
                            ).get("arguments", {})
                            tool_output = tool_call.get("output") or tool_call.get(
                                "result", ""
                            )

                            if tool_name:
                                used_tools.add(tool_name)
                                tool_calls.append(
                                    {
                                        "tool": tool_name,
                                        "input": tool_input,
                                        "output": tool_output,
                                        "timestamp": time.time(),
                                    }
                                )

                                # 发送工具调用事件
                                tool_call_data = {
                                    "type": "tool_call",
                                    "tool": tool_name,
                                    "input": tool_input,
                                    "output": tool_output,
                                }
                                yield f"data: {json.dumps(tool_call_data, ensure_ascii=False)}\n\n"

                # 提取 model_service_info 中的信息
                model_info = extra.get("model_service_info", {})
                output = model_info.get("output", {})
                choices = output.get("choices", [])
                finish_reason = None
                usage = model_info.get("usage", {})

                # 获取 request_id（第一次时设置）
                if not request_id:
                    request_id = model_info.get("request_id", "")

                if choices:
                    finish_reason = choices[0].get("finish_reason")

                # 计算增量内容（delta）
                # 如果当前内容包含上一次的内容，提取增量部分
                if last_content and current_content.startswith(last_content):
                    delta = current_content[len(last_content) :]
                elif not last_content:
                    # 第一次返回，delta 就是完整内容
                    delta = current_content
                else:
                    # 如果内容不连续（不应该发生，但做容错处理）
                    delta = current_content
                    last_content = ""

                # 只有当 delta 不为空时才发送（避免发送空内容）
                if delta or finish_reason:
                    # 构建流式响应数据
                    response_data = {
                        "id": request_id or f"chatcmpl-{created_time}",
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": "qwen-max",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": (
                                        role if not last_content else None
                                    ),  # 只在第一次发送 role
                                    "content": delta,  # 增量内容
                                },
                                "finish_reason": finish_reason,
                            }
                        ],
                    }

                    # 如果需要完整内容（用于调试或兼容），可以添加
                    if current_content:
                        response_data["full_content"] = current_content

                    # 如果有 usage 信息，添加到最后一条消息
                    if usage:
                        response_data["usage"] = usage

                    # Server-Sent Events 格式
                    yield f"data: {json.dumps(response_data, ensure_ascii=False)}\n\n"

                # 更新上一次的内容
                last_content = current_content

        # 发送最终完成消息（包含 finish_reason 和 usage）
        if last_full_response:
            items = (
                last_full_response
                if isinstance(last_full_response, list)
                else [last_full_response]
            )
            for item in items:
                extra = item.get("extra", {})
                model_info = extra.get("model_service_info", {})
                output = model_info.get("output", {})
                choices = output.get("choices", [])
                usage = model_info.get("usage", {})
                finish_reason = (
                    "stop"
                    if choices and choices[0].get("finish_reason") == "stop"
                    else "stop"
                )

                final_data = {
                    "id": request_id
                    or model_info.get("request_id", f"chatcmpl-{created_time}"),
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": "qwen-max",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }

                # 添加 usage 信息到最终消息
                if usage:
                    final_data["usage"] = usage

                # 添加工具调用摘要
                if tool_calls:
                    final_data["tool_calls"] = tool_calls
                    final_data["used_tools"] = list(used_tools)

                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

        # 发送结束标记
        yield "data: [DONE]\n\n"

    except Exception as e:
        # 详细的错误处理
        error_type = type(e).__name__
        error_message = str(e)

        # 特殊处理代理错误
        if "ProxyError" in error_type or "proxy" in error_message.lower():
            error_message = (
                f"代理连接失败: {error_message}\n"
                f"提示: 请检查代理设置，或设置环境变量 DASHSCOPE_DISABLE_PROXY=true 来禁用代理"
            )

        error_data = {
            "error": {
                "message": error_message,
                "type": error_type,
                "code": 500,
            },
            "object": "error",
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


def app_tui():
    """终端交互模式

    提供命令行交互界面，支持：
    - 连续对话
    - 文件输入
    - 实时响应
    """
    try:
        # 初始化助手
        bot = init_agent_service()

        # 对话历史
        messages = []
        while True:
            try:
                # 获取用户输入
                query = input("user question: ")
                # 获取可选的文件输入
                file = input("file url (press enter if no file): ").strip()

                # 输入验证
                if not query:
                    print("user question cannot be empty！")
                    continue

                # 构建消息
                if not file:
                    messages.append({"role": "user", "content": query})
                else:
                    messages.append(
                        {"role": "user", "content": [{"text": query}, {"file": file}]}
                    )

                print("正在处理您的请求...")
                # 运行助手并处理响应
                response = []
                for response in bot.run(messages):
                    print("bot response:", response)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")


def app_gui():
    """图形界面模式

    提供 Web 图形界面，特点：
    - 友好的用户界面
    - 预设查询建议
    - 智能路线规划
    """
    # 初始化助手
    bot = init_agent_service()
    # 配置聊天界面
    chatbot_config = {
        "prompt.suggestions": [
            "帮我规划上海一日游行程，主要想去外滩和迪士尼",
            "我在南京路步行街，帮我找一家评分高的本帮菜餐厅",
            "从浦东机场到外滩怎么走最方便？",
            "推荐上海三个适合拍照的网红景点",
            "帮我查找上海科技馆的具体地址和营业时间",
            "从徐家汇到外滩有哪些公交路线？",
            "现在在豫园，附近有什么好玩的地方推荐？",
            "帮我找一下静安寺附近的停车场",
            "上海野生动物园到迪士尼乐园怎么走？",
            "推荐陆家嘴附近的高档餐厅",
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == "__main__":
    app_gui()
    # app_tui()
