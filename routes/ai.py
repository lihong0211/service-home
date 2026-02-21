# routes/ai.py
"""AI 相关路由：ping、对话、STT、TTS、图像、视频、知识库、向量库、RAG、文件、LangGraph。"""

from flask import jsonify, request

from service.ai.chat import chat, ocr_chat
from service.ai.langchain import (
    get_graph_schema,
    list_graph_names,
    run_graph_and_collect_steps,
)
from service.ai.agent import (
    list_agents,
    get_agent_schema,
    run_agent_and_collect_steps,
)
from service.ai.mcp import (
    mcp_gaode_info_api,
    mcp_gaode_chat_stream_api,
    mcp_ppt_info_api,
    mcp_ppt_chat_api,
    mcp_ppt_chat_stream_api,
    mcp_weather_info_api,
    mcp_weather_chat_api,
    mcp_weather_chat_stream_api,
    mcp_tts_info_api,
    mcp_tts_chat_api,
    mcp_tts_chat_stream_api,
    mcp_stt_info_api,
    mcp_stt_chat_api,
    mcp_stt_chat_stream_api,
)
from service.ai.function_call import (
    function_calling_info_api,
    function_calling_chat_api,
)
from service.ai.stt import (
    transcribe as stt_transcribe,
    transcribe_stream as stt_transcribe_stream,
)
from service.ai.tts import speech as tts_speech
from service.ai.image_gen import generate as image_generate
from service.ai.video_undstanding import video_understand
from service.ai.knowledge import (
    list_knowledge_bases_api,
    create_knowledge_base_api,
    create_knowledge_base_from_pdf_api,
    upload_knowledge_base_api,
    list_knowledge_base_documents_api,
    get_document_segments_api,
    preview_knowledge_document_api,
    preview_segments_from_db_api,
    execute_segments_api,
    sync_knowledge_base_from_disk_api,
    update_knowledge_base_api,
    rebuild_knowledge_base_api,
    vectorize_knowledge_base_api,
    vectorize_with_file_api,
    get_knowledge_base_detail_api,
    delete_knowledge_base_api,
    delete_knowledge_base_document_api,
)
from service.ai.rag import rag_ask_api, rag_search_api
from service.ai.text2sql import text2sql_api, table_data_api
from service.ai.files import upload_file_api, list_files_api, preview_file_api
from service.ai.vector_db import (
    list_api as vector_db_list_api,
    create_api as vector_db_create_api,
    detail_api as vector_db_detail_api,
    update_api as vector_db_update_api,
    update_meta_api as vector_db_update_meta_api,
    delete_api as vector_db_delete_api,
    sync_from_disk_api as vector_db_sync_from_disk_api,
    rebuild_api as vector_db_rebuild_api,
    documents_api as vector_db_documents_api,
    document_add_api as vector_db_document_add_api,
    document_update_api as vector_db_document_update_api,
    document_delete_api as vector_db_document_delete_api,
    categories_api as vector_db_categories_api,
    category_add_api as vector_db_category_add_api,
    category_update_api as vector_db_category_update_api,
    category_delete_api as vector_db_category_delete_api,
    search_api as vector_db_search_api,
)
from service.ai.a2a import get_result_for_frontend as a2a_chain_get_result


def ai_ping():
    """跨域测试：GET /ai/ping"""
    return jsonify({"ok": True})


def langgraph_graph_api():
    """GET /ai/langgraph/graph?name=router 返回图结构，供前端 3D 可视化（GraphData）。"""
    name = request.args.get("name") or "router"
    schema = get_graph_schema(name)
    if schema is None:
        return (
            jsonify(
                {
                    "code": 400,
                    "msg": f"未知图: {name}",
                    "data": {"allowed": list_graph_names()},
                }
            ),
            400,
        )
    return jsonify({"code": 0, "msg": "ok", "data": schema})


def langgraph_run_api():
    """POST /ai/langgraph/run 执行图并返回步骤与最终状态，供前端按真实执行顺序驱动 3D 动画。"""
    body = request.get_json() or {}
    graph_name = body.get("graph") or "router"
    input_state = body.get("input")
    out = run_graph_and_collect_steps(graph_name, input_state)
    if out.get("error"):
        return jsonify({"code": 400, "msg": out["error"], "data": out}), 400
    return jsonify({"code": 0, "msg": "ok", "data": out})


def agent_list_api():
    """GET /ai/agent/list 返回所有可用的智能体列表（含元信息）。"""
    agents = list_agents()
    return jsonify({"code": 0, "msg": "ok", "data": agents})


def agent_schema_api():
    """GET /ai/agent/schema?agent_id=research_agent 返回智能体的图结构，供前端 3D 可视化。"""
    agent_id = request.args.get("agent_id") or "research_agent"
    schema = get_agent_schema(agent_id)
    if schema is None:
        return (
            jsonify(
                {
                    "code": 400,
                    "msg": f"未知智能体: {agent_id}",
                    "data": {"allowed": list(list_agents().keys())},
                }
            ),
            400,
        )
    return jsonify({"code": 0, "msg": "ok", "data": schema})


def agent_run_api():
    """POST /ai/agent/run 执行智能体并返回步骤与最终状态，供前端按真实执行顺序驱动 3D 动画。"""
    body = request.get_json() or {}
    agent_id = body.get("agent_id") or "research_agent"
    input_data = body.get("input")
    out = run_agent_and_collect_steps(agent_id, input_data)
    if out.get("error"):
        return jsonify({"code": 400, "msg": out["error"], "data": out}), 400
    return jsonify({"code": 0, "msg": "ok", "data": out})


def a2a_chain_api():
    """POST /ai/a2a/chain 执行 A2A 内容生成链，返回 chain + artifacts + final_artifact，供前端展示。"""
    body = request.get_json() or {}
    topic = body.get("topic", "").strip()
    if not topic:
        return jsonify({"code": 400, "msg": "缺少参数: topic", "data": None}), 400
    try:
        data = a2a_chain_get_result(topic)
        return jsonify({"code": 0, "msg": "ok", "data": data})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e), "data": None}), 500


def register_ai(bp):
    bp.add_url_rule("/ai/ping", "ai_ping", ai_ping, methods=["GET", "OPTIONS"])
    bp.add_url_rule("/ai/chat", "chat", chat, methods=["POST"])
    bp.add_url_rule("/ai/orc", "ocr_chat", ocr_chat, methods=["POST"])
    bp.add_url_rule(
        "/ai/stt/transcribe", "stt_transcribe", stt_transcribe, methods=["POST"]
    )
    bp.add_url_rule(
        "/ai/stt/transcribe_stream",
        "stt_transcribe_stream",
        stt_transcribe_stream,
        methods=["POST"],
    )
    bp.add_url_rule("/ai/tts", "tts_speech", tts_speech, methods=["POST"])
    bp.add_url_rule(
        "/ai/image/generate", "image_generate", image_generate, methods=["POST"]
    )
    bp.add_url_rule(
        "/ai/video/understand", "video_understand", video_understand, methods=["POST"]
    )

    # 知识库（建库、上传、分段、向量化）
    bp.add_url_rule(
        "/ai/knowledge-base/list",
        "knowledge_base_list",
        list_knowledge_bases_api,
        methods=["GET", "POST"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base",
        "knowledge_base_create",
        create_knowledge_base_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/detail",
        "knowledge_base_detail",
        get_knowledge_base_detail_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/update",
        "knowledge_base_update",
        update_knowledge_base_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/delete",
        "knowledge_base_delete",
        delete_knowledge_base_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/upload",
        "knowledge_base_upload",
        upload_knowledge_base_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/documents",
        "knowledge_base_documents",
        list_knowledge_base_documents_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/document/<int:document_id>/segments",
        "knowledge_base_document_segments",
        get_document_segments_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/document/<int:document_id>/preview",
        "knowledge_base_document_preview",
        preview_knowledge_document_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/document/delete",
        "knowledge_base_document_delete",
        delete_knowledge_base_document_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/from-pdf",
        "knowledge_base_from_pdf",
        create_knowledge_base_from_pdf_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/sync-from-disk",
        "knowledge_base_sync_from_disk",
        sync_knowledge_base_from_disk_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/rebuild",
        "knowledge_base_rebuild",
        rebuild_knowledge_base_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/vectorize",
        "knowledge_base_vectorize",
        vectorize_knowledge_base_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/segments/preview",
        "knowledge_base_segments_preview",
        preview_segments_from_db_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/segments/execute",
        "knowledge_base_segments_execute",
        execute_segments_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/knowledge-base/vectorize-with-file",
        "knowledge_base_vectorize_with_file",
        vectorize_with_file_api,
        methods=["POST"],
    )

    # 向量库直连（文档/分类 CRUD、检索）
    bp.add_url_rule(
        "/ai/vector-db/list",
        "vector_db_list",
        vector_db_list_api,
        methods=["GET", "POST"],
    )
    bp.add_url_rule(
        "/ai/vector-db", "vector_db_create", vector_db_create_api, methods=["POST"]
    )
    bp.add_url_rule(
        "/ai/vector-db/detail",
        "vector_db_detail",
        vector_db_detail_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/vector-db/update",
        "vector_db_update",
        vector_db_update_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/vector-db/update-meta",
        "vector_db_update_meta",
        vector_db_update_meta_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/vector-db/delete",
        "vector_db_delete",
        vector_db_delete_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/vector-db/sync-from-disk",
        "vector_db_sync_from_disk",
        vector_db_sync_from_disk_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/vector-db/rebuild",
        "vector_db_rebuild",
        vector_db_rebuild_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/vector-db/documents",
        "vector_db_documents",
        vector_db_documents_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/vector-db/document/add",
        "vector_db_document_add",
        vector_db_document_add_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/vector-db/document/update",
        "vector_db_document_update",
        vector_db_document_update_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/vector-db/document/delete",
        "vector_db_document_delete",
        vector_db_document_delete_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/vector-db/categories",
        "vector_db_categories",
        vector_db_categories_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/vector-db/category/add",
        "vector_db_category_add",
        vector_db_category_add_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/vector-db/category/update",
        "vector_db_category_update",
        vector_db_category_update_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/vector-db/category/delete",
        "vector_db_category_delete",
        vector_db_category_delete_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/vector-db/search",
        "vector_db_search",
        vector_db_search_api,
        methods=["POST"],
    )

    # RAG
    bp.add_url_rule("/ai/rag/ask", "rag_ask", rag_ask_api, methods=["POST"])
    bp.add_url_rule("/ai/rag/search", "rag_search", rag_search_api, methods=["POST"])

    # Text2SQL（ai 库表：自然语言转 SQL，返回改写后的 SQL 与执行结果）
    bp.add_url_rule("/ai/text2sql", "text2sql", text2sql_api, methods=["POST"])
    # 原始数据：按表名 + 分页查询，返回该表所有字段
    bp.add_url_rule(
        "/ai/table-data", "table_data", table_data_api, methods=["GET", "POST"]
    )

    # 文件（上传、列表、预览）
    bp.add_url_rule(
        "/ai/files/upload", "files_upload", upload_file_api, methods=["POST"]
    )
    bp.add_url_rule("/ai/files/list", "files_list", list_files_api, methods=["GET"])
    bp.add_url_rule(
        "/ai/files/<file_id>/preview",
        "files_preview",
        preview_file_api,
        methods=["GET"],
    )

    # LangGraph（供前端 React+Three.js 3D 可视化）
    bp.add_url_rule(
        "/ai/langgraph/graph", "langgraph_graph", langgraph_graph_api, methods=["GET"]
    )
    bp.add_url_rule(
        "/ai/langgraph/run", "langgraph_run", langgraph_run_api, methods=["POST"]
    )

    # Agent 智能体系统（供前端 React+Three.js 3D 可视化）
    bp.add_url_rule("/ai/agent/list", "agent_list", agent_list_api, methods=["GET"])
    bp.add_url_rule(
        "/ai/agent/schema", "agent_schema", agent_schema_api, methods=["GET"]
    )
    bp.add_url_rule("/ai/agent/run", "agent_run", agent_run_api, methods=["POST"])

    # 高德地图助手（前后端分离，可定制前端）
    bp.add_url_rule(
        "/ai/mcp-gaode/info", "mcp_gaode_info", mcp_gaode_info_api, methods=["GET"]
    )
    bp.add_url_rule(
        "/ai/mcp-gaode/chat-stream",
        "mcp_gaode_chat_stream",
        mcp_gaode_chat_stream_api,
        methods=["POST"],
    )

    # Function Calling 助手（前后端分离，可定制前端）
    bp.add_url_rule(
        "/ai/function-calling/info",
        "function_calling_info",
        function_calling_info_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/function-calling/chat",
        "function_calling_chat",
        function_calling_chat_api,
        methods=["POST"],
    )

    # PPT 汇报助手（MCP 能力：拉取数据 + 生成 PPT，可接入活字格/百炼等）
    bp.add_url_rule(
        "/ai/mcp-ppt/info",
        "mcp_ppt_info",
        mcp_ppt_info_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/mcp-ppt/chat",
        "mcp_ppt_chat",
        mcp_ppt_chat_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/mcp-ppt/chat-stream",
        "mcp_ppt_chat_stream",
        mcp_ppt_chat_stream_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/mcp-weather/info",
        "mcp_weather_info",
        mcp_weather_info_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/mcp-weather/chat",
        "mcp_weather_chat",
        mcp_weather_chat_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/mcp-weather/chat-stream",
        "mcp_weather_chat_stream",
        mcp_weather_chat_stream_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/mcp-tts/info",
        "mcp_tts_info",
        mcp_tts_info_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/mcp-tts/chat",
        "mcp_tts_chat",
        mcp_tts_chat_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/mcp-tts/chat-stream",
        "mcp_tts_chat_stream",
        mcp_tts_chat_stream_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/mcp-stt/info",
        "mcp_stt_info",
        mcp_stt_info_api,
        methods=["GET"],
    )
    bp.add_url_rule(
        "/ai/mcp-stt/chat",
        "mcp_stt_chat",
        mcp_stt_chat_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/mcp-stt/chat-stream",
        "mcp_stt_chat_stream",
        mcp_stt_chat_stream_api,
        methods=["POST"],
    )
    bp.add_url_rule(
        "/ai/a2a/chain",
        "a2a_chain",
        a2a_chain_api,
        methods=["POST"],
    )
