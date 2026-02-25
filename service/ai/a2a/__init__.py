# A2A 包：标准 A2A 协议实现，供 routes/ai 注册 POST /ai/a2a/chain
from service.ai.a2a.orchestrator import (
    get_result_for_frontend,
    a2a_chain_api,
    a2a_chain_stream_api,
)

__all__ = ["get_result_for_frontend", "a2a_chain_api", "a2a_chain_stream_api"]
