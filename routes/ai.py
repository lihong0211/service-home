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
from service.ai.a2a import a2a_chain_api, a2a_chain_stream_api
from service.ai.finetuning.finetuning import finetuning_chat_api, list_lora_options_api
from service.ai.docs import service_ai_doc_api
from service.ai.agent.agent_doctor import doctor_chat_api, doctor_session_api


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
        async def handler(request: Request, db: SessionDep, **kwargs):
            set_request_session(db)
            try:
                conv = {
                    k: (param_types[k](v) if param_types and k in param_types else v)
                    for k, v in kwargs.items()
                }
                result = await _dispatch_ai_view(view, request, **conv)
                return normalize_api_result(result)
            finally:
                clear_request_session()

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
