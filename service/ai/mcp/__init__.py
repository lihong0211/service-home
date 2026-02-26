# MCP 包：高德、PPT、天气、TTS、STT 等 MCP 助手，供 routes/ai 注册
from service.ai.mcp.mcp_gaode import (
    mcp_gaode_info_api,
    mcp_gaode_chat_stream_api,
)
from service.ai.mcp.mcp_ppt import (
    mcp_ppt_info_api,
    mcp_ppt_chat_api,
    mcp_ppt_chat_stream_api,
    mcp_ppt_status_api,
    mcp_ppt_download_url_api,
    mcp_ppt_download_proxy_api,
    mcp_ppt_editor_url_api,
    mcp_ppt_history_api,
)
from service.ai.mcp.mcp_weather import (
    mcp_weather_info_api,
    mcp_weather_chat_api,
    mcp_weather_chat_stream_api,
)
from service.ai.mcp.mcp_tts import (
    mcp_tts_info_api,
    mcp_tts_chat_api,
    mcp_tts_chat_stream_api,
)
from service.ai.mcp.mcp_stt import (
    mcp_stt_info_api,
    mcp_stt_chat_api,
    mcp_stt_chat_stream_api,
)

__all__ = [
    "mcp_gaode_info_api",
    "mcp_gaode_chat_stream_api",
    "mcp_ppt_info_api",
    "mcp_ppt_chat_api",
    "mcp_ppt_chat_stream_api",
    "mcp_ppt_status_api",
    "mcp_ppt_download_url_api",
    "mcp_ppt_download_proxy_api",
    "mcp_ppt_editor_url_api",
    "mcp_ppt_history_api",
    "mcp_weather_info_api",
    "mcp_weather_chat_api",
    "mcp_weather_chat_stream_api",
    "mcp_tts_info_api",
    "mcp_tts_chat_api",
    "mcp_tts_chat_stream_api",
    "mcp_stt_info_api",
    "mcp_stt_chat_api",
    "mcp_stt_chat_stream_api",
]
