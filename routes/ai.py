# routes/ai.py
"""AI 相关路由：ping、对话、STT、TTS、图像、视频、知识库、向量库、RAG、文件、LangGraph。"""
from __future__ import annotations

import inspect

import anyio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.database import clear_request_session, set_request_session
from app.deps import SessionDep
from utils.api_result import normalize_api_result

from service.ai.chat import chat, ocr_chat
from service.ai.langchain import langgraph_graph_api, langgraph_run_api
from service.ai.agent import (
    agent_list_api,
    agent_schema_api,
    agent_run_api,
)
from service.ai.mcp import (
    mcp_gaode_info_api,
    mcp_gaode_chat_stream_api,
    mcp_ppt_info_api,
    mcp_ppt_chat_api,
    mcp_ppt_chat_stream_api,
    mcp_ppt_status_api,
    mcp_ppt_download_url_api,
    mcp_ppt_download_proxy_api,
    mcp_ppt_editor_url_api,
    mcp_ppt_history_api,
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
from service.ai.video_gen_task import (
    video_gen_task_create_api,
    video_gen_task_get_api,
    video_gen_task_list_api,
)
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
from service.ai.bm25_es import bm25_sync_api
from service.ai.text2sql import text2sql_api, table_data_api
from service.ai.files import upload_file_api, list_files_api, preview_file_api
from service.ai.vector_db_qdrant import (
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
from service.ai.a2a import a2a_chain_api, a2a_chain_stream_api
from service.ai.finetuning.finetuning import finetuning_chat_api, list_lora_options_api
from service.ai.docs import service_ai_doc_api
from service.ai.agent.agent_doctor import doctor_chat_api, doctor_session_api
from service.ai.data_analysis import upload_data_file_api, query_data_api
from service.ai.github_chat import github_index_api, github_ask_api
from service.ai.youtube_chat import youtube_index_api, youtube_ask_api
from service.ai.memory_chat import memory_chat_api, list_memories_api, clear_memories_api
from service.ai.mixture_agents import list_models_api, mixture_chat_api
from service.ai.resume_matcher import resume_match_api
from service.ai.news_agent import fetch_articles_api, news_summary_api
from service.ai.web_scraper import web_scrape_extract_api
# starter_agents
from service.ai.starter_agents.travel_agent import travel_plan_api
from service.ai.starter_agents.recipe_agent import recipe_plan_api
from service.ai.starter_agents.health_fitness_agent import health_plan_api
from service.ai.starter_agents.reasoning_agent import reasoning_chat_api
from service.ai.starter_agents.finance_coach import finance_plan_api
from service.ai.starter_agents.mental_wellbeing import wellbeing_chat_api
from service.ai.starter_agents.startup_trend import startup_analyze_api
# advanced_agents
from service.ai.advanced_agents.speech_trainer import speech_analyze_api
from service.ai.advanced_agents.negotiation_simulator import negotiation_chat_api, list_scenarios_api
from service.ai.advanced_agents.chess_game import chess_new_api, chess_move_api
# chat_with_x
from service.ai.chat_with_x.pdf_chat import pdf_index_api, pdf_ask_api
from service.ai.chat_with_x.arxiv_chat import arxiv_index_api, arxiv_ask_api
from service.ai.chat_with_x.gmail_chat import (
    gmail_auth_api, gmail_callback_api, gmail_list_api, gmail_summarize_api, gmail_reply_draft_api,
)
# llm_apps
from service.ai.llm_apps.blog_podcast import blog_script_api, blog_to_podcast_api
from service.ai.llm_apps.data_viz import data_viz_columns_api, data_viz_api
from service.ai.llm_apps.tarot_chat import tarot_read_api
from service.ai.llm_apps.music_gen_agent import music_generate_api, music_status_api
# moss-tts
from service.ai.moss_tts import moss_tts_speech_api, moss_tts_status_api


async def _dispatch_ai_view(view, request: Request, **path_kwargs):
    """调用业务视图：统一传入 Starlette Request；同步函数放到线程池。"""
    if inspect.iscoroutinefunction(view):
        return await view(request, **path_kwargs) if path_kwargs else await view(request)

    def _call():
        return view(request, **path_kwargs) if path_kwargs else view(request)

    return await anyio.to_thread.run_sync(_call)


def _ai_route(
    router: APIRouter,
    path: str,
    view,
    methods: list,
    path_params: list | None = None,
    param_types: dict | None = None,
):
    """注册 AI 业务视图；DB 通过 SessionDep 注入并写入 ContextVar 供 db.session 使用。"""
    if path_params:
        # FastAPI 不会把路径参数灌进 **kwargs，会把 kwargs 当成 Query 字段 → 422。
        # 必须为每个 {path_var} 生成显式形参（仅允许 int/str，来自本文件注册表）。
        param_types = param_types or {}
        ann_parts: list[str] = []
        for name in path_params:
            t = param_types.get(name, str)
            if t is int:
                ann_parts.append(f"{name}: int")
            else:
                ann_parts.append(f"{name}: str")
        ann_sig = ", ".join(ann_parts)
        conv_literal = ", ".join(f'"{p}": {p}' for p in path_params)
        # 闭包捕获 view；exec 仅用于生成带正确签名的 async 函数，便于 OpenAPI/依赖注入识别 Path
        src = f"""async def handler(request: Request, db: SessionDep, {ann_sig}):
    set_request_session(db)
    try:
        conv = {{{conv_literal}}}
        result = await _dispatch_ai_view(view, request, **conv)
        return normalize_api_result(result)
    finally:
        clear_request_session()
"""
        namespace = {
            "Request": Request,
            "SessionDep": SessionDep,
            "set_request_session": set_request_session,
            "clear_request_session": clear_request_session,
            "_dispatch_ai_view": _dispatch_ai_view,
            "normalize_api_result": normalize_api_result,
            "view": view,
        }
        exec(src, namespace)
        handler = namespace["handler"]
        handler.__name__ = f"wrap_{getattr(view, '__name__', 'unknown')}"
        router.add_api_route(path, handler, methods=methods)
    else:
        async def handler(request: Request, db: SessionDep):
            set_request_session(db)
            try:
                result = await _dispatch_ai_view(view, request)
                return normalize_api_result(result)
            finally:
                clear_request_session()

        handler.__name__ = f"wrap_{getattr(view, '__name__', 'unknown')}"
        router.add_api_route(path, handler, methods=methods)


def register_ai(router: APIRouter):
    _ai_route(router, "/ai/chat", chat, ["POST"])
    _ai_route(router, "/ai/orc", ocr_chat, ["POST"])

    @router.post("/ai/stt/transcribe")
    async def stt_transcribe_route(request: Request, db: SessionDep):
        set_request_session(db)
        try:
            result = await stt_transcribe(request)
            if isinstance(result, tuple):
                body, status_code = result
            else:
                body, status_code = result, 200
            return JSONResponse(content=body, status_code=status_code)
        finally:
            clear_request_session()

    @router.post("/ai/stt/transcribe_stream")
    async def stt_transcribe_stream_route(request: Request, db: SessionDep):
        set_request_session(db)
        try:
            return await stt_transcribe_stream(request)
        finally:
            clear_request_session()

    @router.post("/ai/tts")
    async def tts_speech_route(request: Request, db: SessionDep):
        set_request_session(db)
        try:
            return await tts_speech(request)
        finally:
            clear_request_session()

    _ai_route(router, "/ai/image/generate", image_generate, ["POST"])
    _ai_route(router, "/ai/video/understand", video_understand, ["POST"])
    _ai_route(router, "/ai/video-gen/tasks", video_gen_task_create_api, ["POST"])
    _ai_route(router, "/ai/video-gen/tasks/{task_id}", video_gen_task_get_api, ["GET"], ["task_id"])
    _ai_route(router, "/ai/video-gen/tasks/list", video_gen_task_list_api, ["GET"])

    # 知识库
    _ai_route(router, "/ai/knowledge-base/list", list_knowledge_bases_api, ["GET", "POST"])
    _ai_route(router, "/ai/knowledge-base", create_knowledge_base_api, ["POST"])
    _ai_route(router, "/ai/knowledge-base/detail", get_knowledge_base_detail_api, ["GET"])
    _ai_route(router, "/ai/knowledge-base/update", update_knowledge_base_api, ["POST"])
    _ai_route(router, "/ai/knowledge-base/delete", delete_knowledge_base_api, ["POST"])
    _ai_route(router, "/ai/knowledge-base/upload", upload_knowledge_base_api, ["POST"])
    _ai_route(router, "/ai/knowledge-base/documents", list_knowledge_base_documents_api, ["GET"])
    _ai_route(router, "/ai/knowledge-base/document/{document_id}/segments", get_document_segments_api, ["GET"], ["document_id"], {"document_id": int})
    _ai_route(router, "/ai/knowledge-base/document/{document_id}/preview", preview_knowledge_document_api, ["GET"], ["document_id"], {"document_id": int})
    _ai_route(router, "/ai/knowledge-base/document/delete", delete_knowledge_base_document_api, ["POST"])
    _ai_route(router, "/ai/knowledge-base/from-pdf", create_knowledge_base_from_pdf_api, ["POST"])
    _ai_route(router, "/ai/knowledge-base/sync-from-disk", sync_knowledge_base_from_disk_api, ["POST"])
    _ai_route(router, "/ai/knowledge-base/rebuild", rebuild_knowledge_base_api, ["POST"])
    _ai_route(router, "/ai/knowledge-base/vectorize", vectorize_knowledge_base_api, ["POST"])
    _ai_route(router, "/ai/knowledge-base/segments/preview", preview_segments_from_db_api, ["GET"])
    _ai_route(router, "/ai/knowledge-base/segments/execute", execute_segments_api, ["POST"])
    _ai_route(router, "/ai/knowledge-base/vectorize-with-file", vectorize_with_file_api, ["POST"])

    # 向量库
    _ai_route(router, "/ai/vector-db/list", vector_db_list_api, ["GET", "POST"])
    _ai_route(router, "/ai/vector-db", vector_db_create_api, ["POST"])
    _ai_route(router, "/ai/vector-db/detail", vector_db_detail_api, ["GET"])
    _ai_route(router, "/ai/vector-db/update", vector_db_update_api, ["POST"])
    _ai_route(router, "/ai/vector-db/update-meta", vector_db_update_meta_api, ["POST"])
    _ai_route(router, "/ai/vector-db/delete", vector_db_delete_api, ["POST"])
    _ai_route(router, "/ai/vector-db/sync-from-disk", vector_db_sync_from_disk_api, ["POST"])
    _ai_route(router, "/ai/vector-db/rebuild", vector_db_rebuild_api, ["POST"])
    _ai_route(router, "/ai/vector-db/documents", vector_db_documents_api, ["GET"])
    _ai_route(router, "/ai/vector-db/document/add", vector_db_document_add_api, ["POST"])
    _ai_route(router, "/ai/vector-db/document/update", vector_db_document_update_api, ["POST"])
    _ai_route(router, "/ai/vector-db/document/delete", vector_db_document_delete_api, ["POST"])
    _ai_route(router, "/ai/vector-db/categories", vector_db_categories_api, ["GET"])
    _ai_route(router, "/ai/vector-db/category/add", vector_db_category_add_api, ["POST"])
    _ai_route(router, "/ai/vector-db/category/update", vector_db_category_update_api, ["POST"])
    _ai_route(router, "/ai/vector-db/category/delete", vector_db_category_delete_api, ["POST"])
    _ai_route(router, "/ai/vector-db/search", vector_db_search_api, ["POST"])

    # BM25（Elasticsearch，可选组件）
    # 仅提供同步接口：检索只在 RAG 内部作为兜底启用（enable_bm25=true）
    _ai_route(router, "/ai/bm25/sync", bm25_sync_api, ["POST"])

    _ai_route(router, "/ai/rag/ask", rag_ask_api, ["POST"])
    _ai_route(router, "/ai/rag/search", rag_search_api, ["POST"])
    _ai_route(router, "/ai/text2sql", text2sql_api, ["POST"])
    _ai_route(router, "/ai/table-data", table_data_api, ["GET", "POST"])
    _ai_route(router, "/ai/files/upload", upload_file_api, ["POST"])
    _ai_route(router, "/ai/files/list", list_files_api, ["GET"])
    _ai_route(router, "/ai/files/{file_id}/preview", preview_file_api, ["GET"], ["file_id"])

    _ai_route(router, "/ai/langgraph/graph", langgraph_graph_api, ["GET"])
    _ai_route(router, "/ai/langgraph/run", langgraph_run_api, ["POST"])
    _ai_route(router, "/ai/agent/list", agent_list_api, ["GET"])
    _ai_route(router, "/ai/agent/schema", agent_schema_api, ["GET"])
    _ai_route(router, "/ai/agent/run", agent_run_api, ["POST"])

    _ai_route(router, "/ai/mcp-gaode/info", mcp_gaode_info_api, ["GET"])
    _ai_route(router, "/ai/mcp-gaode/chat-stream", mcp_gaode_chat_stream_api, ["POST"])
    _ai_route(router, "/ai/function-calling/info", function_calling_info_api, ["GET"])
    _ai_route(router, "/ai/function-calling/chat", function_calling_chat_api, ["POST"])
    _ai_route(router, "/ai/mcp-ppt/info", mcp_ppt_info_api, ["GET"])
    _ai_route(router, "/ai/mcp-ppt/chat", mcp_ppt_chat_api, ["POST"])
    _ai_route(router, "/ai/mcp-ppt/chat-stream", mcp_ppt_chat_stream_api, ["POST"])
    _ai_route(router, "/ai/mcp-ppt/status", mcp_ppt_status_api, ["GET"])
    _ai_route(router, "/ai/mcp-ppt/download-url", mcp_ppt_download_url_api, ["GET"])
    _ai_route(router, "/ai/mcp-ppt/download", mcp_ppt_download_proxy_api, ["GET"])
    _ai_route(router, "/ai/mcp-ppt/editor", mcp_ppt_editor_url_api, ["GET"])
    _ai_route(router, "/ai/mcp-ppt/history", mcp_ppt_history_api, ["GET"])
    _ai_route(router, "/ai/mcp-weather/info", mcp_weather_info_api, ["GET"])
    _ai_route(router, "/ai/mcp-weather/chat", mcp_weather_chat_api, ["POST"])
    _ai_route(router, "/ai/mcp-weather/chat-stream", mcp_weather_chat_stream_api, ["POST"])
    _ai_route(router, "/ai/mcp-tts/info", mcp_tts_info_api, ["GET"])
    _ai_route(router, "/ai/mcp-tts/chat", mcp_tts_chat_api, ["POST"])
    _ai_route(router, "/ai/mcp-tts/chat-stream", mcp_tts_chat_stream_api, ["POST"])
    _ai_route(router, "/ai/mcp-stt/info", mcp_stt_info_api, ["GET"])
    _ai_route(router, "/ai/mcp-stt/chat", mcp_stt_chat_api, ["POST"])
    _ai_route(router, "/ai/mcp-stt/chat-stream", mcp_stt_chat_stream_api, ["POST"])
    _ai_route(router, "/ai/a2a/chain", a2a_chain_api, ["POST"])
    _ai_route(router, "/ai/a2a/chain/stream", a2a_chain_stream_api, ["POST"])

    _ai_route(router, "/ai/finetuning/chat", finetuning_chat_api, ["POST"])
    _ai_route(router, "/ai/finetuning/lora-options", list_lora_options_api, ["GET"])
    _ai_route(router, "/ai/doctor/chat", doctor_chat_api, ["POST"])
    _ai_route(router, "/ai/doctor/session/{session_id}", doctor_session_api, ["GET"], ["session_id"])
    _ai_route(router, "/ai/docs/{doc_id}", service_ai_doc_api, ["GET"], ["doc_id"], {"doc_id": int})
    _ai_route(router, "/ai/data-analysis/upload", upload_data_file_api, ["POST"])
    _ai_route(router, "/ai/data-analysis/query", query_data_api, ["POST"])
    _ai_route(router, "/ai/github-chat/index", github_index_api, ["POST"])
    _ai_route(router, "/ai/github-chat/ask", github_ask_api, ["POST"])
    _ai_route(router, "/ai/youtube-chat/index", youtube_index_api, ["POST"])
    _ai_route(router, "/ai/youtube-chat/ask", youtube_ask_api, ["POST"])
    _ai_route(router, "/ai/memory-chat/chat", memory_chat_api, ["POST"])
    _ai_route(router, "/ai/memory-chat/memories", list_memories_api, ["GET"])
    _ai_route(router, "/ai/memory-chat/memories", clear_memories_api, ["DELETE"])
    _ai_route(router, "/ai/mixture-agents/models", list_models_api, ["GET"])
    _ai_route(router, "/ai/mixture-agents/chat", mixture_chat_api, ["POST"])
    _ai_route(router, "/ai/resume-match", resume_match_api, ["POST"])
    _ai_route(router, "/ai/news/articles", fetch_articles_api, ["GET"])
    _ai_route(router, "/ai/news/summary", news_summary_api, ["GET"])
    _ai_route(router, "/ai/web-scraper/extract", web_scrape_extract_api, ["POST"])
    # starter_agents
    _ai_route(router, "/ai/travel-agent/plan", travel_plan_api, ["POST"])
    _ai_route(router, "/ai/recipe-agent/plan", recipe_plan_api, ["POST"])
    _ai_route(router, "/ai/health-agent/plan", health_plan_api, ["POST"])
    _ai_route(router, "/ai/reasoning-agent/chat", reasoning_chat_api, ["POST"])
    _ai_route(router, "/ai/finance-coach/plan", finance_plan_api, ["POST"])
    _ai_route(router, "/ai/wellbeing/chat", wellbeing_chat_api, ["POST"])
    _ai_route(router, "/ai/startup-trend/analyze", startup_analyze_api, ["POST"])
    # advanced_agents
    _ai_route(router, "/ai/speech-trainer/analyze", speech_analyze_api, ["POST"])
    _ai_route(router, "/ai/negotiation/chat", negotiation_chat_api, ["POST"])
    _ai_route(router, "/ai/negotiation/scenarios", list_scenarios_api, ["GET"])
    _ai_route(router, "/ai/chess/new", chess_new_api, ["POST"])
    _ai_route(router, "/ai/chess/move", chess_move_api, ["POST"])
    # chat_with_x
    _ai_route(router, "/ai/pdf-chat/index", pdf_index_api, ["POST"])
    _ai_route(router, "/ai/pdf-chat/ask", pdf_ask_api, ["POST"])
    _ai_route(router, "/ai/arxiv-chat/index", arxiv_index_api, ["POST"])
    _ai_route(router, "/ai/arxiv-chat/ask", arxiv_ask_api, ["POST"])
    _ai_route(router, "/ai/gmail/auth", gmail_auth_api, ["GET"])
    _ai_route(router, "/ai/gmail/callback", gmail_callback_api, ["GET"])
    _ai_route(router, "/ai/gmail/list", gmail_list_api, ["GET"])
    _ai_route(router, "/ai/gmail/summarize", gmail_summarize_api, ["POST"])
    _ai_route(router, "/ai/gmail/reply-draft", gmail_reply_draft_api, ["POST"])
    # llm_apps
    _ai_route(router, "/ai/blog-podcast/script", blog_script_api, ["POST"])
    _ai_route(router, "/ai/blog-podcast/audio", blog_to_podcast_api, ["POST"])
    _ai_route(router, "/ai/data-viz/columns", data_viz_columns_api, ["POST"])
    _ai_route(router, "/ai/data-viz/chart", data_viz_api, ["POST"])
    _ai_route(router, "/ai/tarot/read", tarot_read_api, ["POST"])
    _ai_route(router, "/ai/music-gen/generate", music_generate_api, ["POST"])
    _ai_route(router, "/ai/music-gen/status", music_status_api, ["GET"])
    # moss-tts
    _ai_route(router, "/ai/moss-tts/speech", moss_tts_speech_api, ["POST"])
    _ai_route(router, "/ai/moss-tts/status", moss_tts_status_api, ["GET"])
