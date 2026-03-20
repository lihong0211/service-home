"""
医生智能体 - 多轮对话采集患者信息，进行深度诊断分析

架构：LangGraph StateGraph + MemorySaver（多轮会话状态持久化）

流程：
  用户发消息
      ↓
  [extract_info]  从最新消息中提取结构化患者信息，合并到 patient_info
      ↓
  [check_completeness]  判断信息完整度（条件路由）
      ↓                          ↓
  信息不足                    信息充分（或已问足够多轮）
      ↓                          ↓
  [ask_questions]          [generate_assessment]
  生成 1-2 个引导问题           深度诊断分析报告
      ↓                          ↓
     END                        END

每轮对话用相同 session_id 作为 thread_id，MemorySaver 自动保留历史状态。
"""

import json
import os
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from fastapi import Request

from utils.http_body import read_json_optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
_LLM = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=_API_KEY,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3,
    max_tokens=3000,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class DoctorState(TypedDict):
    messages: Annotated[list, add_messages]  # 自动追加，保留完整对话历史
    patient_info: Dict[str, Any]  # 结构化患者信息，逐轮累积
    collection_phase: str  # "collecting" | "completed"
    turn_count: int  # 已问诊轮次
    assessment: Optional[str]  # 最终诊断报告


# ---------------------------------------------------------------------------
# 字段定义与优先级
# ---------------------------------------------------------------------------

_CRITICAL_FIELDS = [
    "age",
    "gender",
    "chief_complaint",
    "symptom_onset",
    "symptom_duration",
    "severity",
    "accompanying_symptoms",
]
_IMPORTANT_FIELDS = [
    "past_medical_history",
    "current_medications",
]
_OPTIONAL_FIELDS = [
    "name",
    "allergies",
    "family_history",
    "aggravating_factors",
    "relieving_factors",
]

_FIELD_LABELS: Dict[str, str] = {
    "name": "姓名",
    "age": "年龄",
    "gender": "性别",
    "chief_complaint": "主要症状/主诉",
    "symptom_onset": "发病时间",
    "symptom_duration": "症状持续时长",
    "severity": "症状严重程度",
    "accompanying_symptoms": "伴随症状",
    "aggravating_factors": "加重因素",
    "relieving_factors": "缓解因素",
    "past_medical_history": "既往病史",
    "current_medications": "当前用药",
    "allergies": "过敏史",
    "family_history": "家族史",
}

_EMPTY_VALUES = {"", "none", "null", "未知", "不详", "无", "没有"}


def _is_filled(value: Any) -> bool:
    return bool(value) and str(value).strip().lower() not in _EMPTY_VALUES


def _get_missing_fields(patient_info: dict) -> List[str]:
    """返回未收集的字段列表，按 critical → important → optional 顺序。"""
    return [
        f
        for f in _CRITICAL_FIELDS + _IMPORTANT_FIELDS + _OPTIONAL_FIELDS
        if not _is_filled(patient_info.get(f))
    ]


def _is_info_sufficient(patient_info: dict, turn_count: int) -> bool:
    """
    判断是否可进入诊断阶段：
    - 所有 critical 字段已填 + important 字段至少 1 个已填，或
    - 已问诊 10 轮以上（避免无限循环）
    """
    critical_ok = all(_is_filled(patient_info.get(f)) for f in _CRITICAL_FIELDS)
    important_filled = sum(
        1 for f in _IMPORTANT_FIELDS if _is_filled(patient_info.get(f))
    )
    return (critical_ok and important_filled >= 1) or turn_count >= 10


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """你是一位专业医生助手。请从患者最新说的话中，提取结构化的患者信息。

当前已收集的患者信息（JSON）：
{current_info}

患者最新说的话：
{latest_message}

请以 JSON 格式**仅返回本轮新提取或更新的字段**（没有新信息则返回空对象 {{}}）。

可提取的字段：
- name（姓名）
- age（年龄，保留原始表达，如 "32岁"）
- gender（性别，"男" 或 "女"）
- chief_complaint（主诉/主要症状，用患者自己的语言）
- symptom_onset（发病时间，如 "昨天下午"、"三天前"）
- symptom_duration（持续时长，如 "2天"、"断断续续一周"）
- severity（严重程度描述，如 "剧烈，影响睡眠"、"轻微"）
- accompanying_symptoms（伴随症状，如 "恶心、低烧"）
- aggravating_factors（使症状加重的因素）
- relieving_factors（使症状缓解的因素）
- past_medical_history（既往病史，如 "高血压病史3年"）
- current_medications（当前用药，如 "氨氯地平5mg每日一次"）
- allergies（过敏史，如 "青霉素过敏"，无则填 "无"）
- family_history（家族病史，如 "父亲有糖尿病"）

只返回 JSON，不要任何解释或多余内容。"""


_ASK_PROMPT = """你是一位经验丰富的全科医生，正在问诊。语气自然、专业，像真实的医生在诊室跟患者说话。

当前已收集的患者信息：
{current_info}

还未收集的信息（按重要性排列）：
{missing_fields}

对话历史：
{history}

根据上面的信息，提出 1-2 个最关键的问题。

要求：
- 语气自然，像真实医生问诊，不生硬，也不过度客套（避免"谢谢您告诉我"、"感谢您"等寒暄）
- 每次只问最重要的 1-2 个问题，不要一口气列出很多，多个问题写在同一段里，不要换行
- 根据上下文灵活判断优先级，如果患者已提到相关信息，顺着追问细节
- 不要重复已经得到答案的问题

直接输出医生说的话，不要任何前缀或后缀。"""


_ASSESS_PROMPT = """你是一位资深全科医生，根据以下患者信息，输出诊断分析报告。

患者信息：
{patient_info}

完整问诊对话：
{history}

严格要求：只能使用患者在对话中**明确说出**的信息。患者未提及的症状、检查结果、用药史等，一律不得出现在报告中，不得推断"无某症状"或"未使用某药物"。

直接输出以下六个部分，不要在报告开头加任何引导语、总结性开场白或情感安慰语：

**一、患者信息摘要**
简要列出核心病史信息（年龄、性别、主诉、病程等）。

**二、症状分析**
对主诉及伴随症状进行详细分析：
- 症状特点与规律（时间、性质、程度、部位等）
- 可能涉及的系统或器官
- 症状之间的关联性

**三、鉴别诊断**
列出 2-4 个最可能的诊断，每个包括：
- 诊断名称及可能性评估（高/中/低）
- 支持该诊断的证据
- 不支持或存疑之处

**四、初步诊断意见**
综合分析后的倾向性诊断及依据。

**五、建议检查项目**
为明确诊断建议进行的检查（血液、影像、其他）。

**六、治疗建议**
- 一般处理（休息、饮食、生活方式）
- 药物治疗建议（如适用，需遵医嘱）
- 是否需要转专科及推荐科室"""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def _msg_content(m: Any) -> str:
    if hasattr(m, "content"):
        return m.content or ""
    if isinstance(m, dict):
        return m.get("content") or ""
    return str(m)


def _build_history_text(messages: list) -> str:
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage) or (
            isinstance(m, dict) and m.get("role") == "user"
        ):
            parts.append(f"患者：{_msg_content(m)}")
        elif isinstance(m, AIMessage) or (
            isinstance(m, dict) and m.get("role") == "assistant"
        ):
            parts.append(f"医生：{_msg_content(m)}")
    return "\n".join(parts) if parts else "（首次问诊）"


def _extract_info(state: DoctorState) -> dict:
    """从最新用户消息中提取结构化患者信息，合并入 patient_info。"""
    messages = state.get("messages", [])
    patient_info = dict(state.get("patient_info") or {})

    # 取最新一条 HumanMessage
    latest_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) or (
            isinstance(m, dict) and m.get("role") == "user"
        ):
            latest_msg = _msg_content(m)
            break

    if latest_msg:
        try:
            chain = (
                ChatPromptTemplate.from_template(_EXTRACT_PROMPT)
                | _LLM
                | JsonOutputParser()
            )
            extracted = chain.invoke(
                {
                    "current_info": json.dumps(
                        patient_info, ensure_ascii=False, indent=2
                    ),
                    "latest_message": latest_msg,
                }
            )
            if isinstance(extracted, dict):
                for k, v in extracted.items():
                    if v is not None and str(v).strip():
                        patient_info[k] = v
        except Exception:
            pass  # 提取失败不中断流程，下轮继续

    return {
        "patient_info": patient_info,
        "turn_count": state.get("turn_count", 0) + 1,
    }


def _check_completeness(
    state: DoctorState,
) -> Literal["ask_questions", "generate_assessment"]:
    """条件路由：信息充分则进入诊断，否则继续问诊。"""
    if _is_info_sufficient(state.get("patient_info") or {}, state.get("turn_count", 0)):
        return "generate_assessment"
    return "ask_questions"


def _ask_questions(state: DoctorState) -> dict:
    """生成 1-2 个自然的引导性问题，推动患者提供关键信息。"""
    patient_info = state.get("patient_info") or {}
    messages = state.get("messages", [])
    missing = _get_missing_fields(patient_info)

    missing_labels = [_FIELD_LABELS.get(f, f) for f in missing[:6]]

    try:
        chain = ChatPromptTemplate.from_template(_ASK_PROMPT) | _LLM | StrOutputParser()
        reply = chain.invoke(
            {
                "current_info": json.dumps(patient_info, ensure_ascii=False, indent=2),
                "missing_fields": (
                    "、".join(missing_labels)
                    if missing_labels
                    else "（基本信息已收集完毕）"
                ),
                "history": _build_history_text(messages),
            }
        )
    except Exception:
        reply = "您好，请问您现在主要哪里不舒服？症状大概从什么时候开始的？"

    return {
        "messages": [AIMessage(content=reply)],
        "collection_phase": "collecting",
    }


def _generate_assessment(state: DoctorState) -> dict:
    """基于完整患者信息，生成深度诊断分析报告。"""
    patient_info = state.get("patient_info") or {}
    messages = state.get("messages", [])

    try:
        chain = (
            ChatPromptTemplate.from_template(_ASSESS_PROMPT) | _LLM | StrOutputParser()
        )
        assessment = chain.invoke(
            {
                "patient_info": json.dumps(patient_info, ensure_ascii=False, indent=2),
                "history": _build_history_text(messages),
            }
        )
    except Exception as e:
        assessment = f"诊断分析生成失败，请稍后重试。（错误：{e}）"

    return {
        "messages": [AIMessage(content=assessment)],
        "collection_phase": "completed",
        "assessment": assessment,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

_memory = MemorySaver()
_doctor_graph = None


def _build_graph():
    workflow = StateGraph(DoctorState)
    workflow.add_node("extract_info", _extract_info)
    workflow.add_node("ask_questions", _ask_questions)
    workflow.add_node("generate_assessment", _generate_assessment)
    workflow.set_entry_point("extract_info")
    workflow.add_conditional_edges(
        "extract_info",
        _check_completeness,
        {
            "ask_questions": "ask_questions",
            "generate_assessment": "generate_assessment",
        },
    )
    workflow.add_edge("ask_questions", END)
    workflow.add_edge("generate_assessment", END)
    return workflow.compile(checkpointer=_memory)


def get_doctor_graph():
    global _doctor_graph
    if _doctor_graph is None:
        _doctor_graph = _build_graph()
    return _doctor_graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ALL_FIELDS = _CRITICAL_FIELDS + _IMPORTANT_FIELDS + _OPTIONAL_FIELDS


def _calc_completion_pct(patient_info: dict) -> int:
    filled = sum(1 for f in _ALL_FIELDS if _is_filled(patient_info.get(f)))
    return round(filled / len(_ALL_FIELDS) * 100)


def chat(session_id: str, message: str) -> dict:
    """
    向医生智能体发送一条消息，返回医生回复及当前问诊状态。

    Args:
        session_id: 会话 ID（同一次问诊保持不变）
        message: 患者发送的消息

    Returns:
        {
            "reply": str,            # 医生回复
            "patient_info": dict,    # 已收集的结构化患者信息
            "completion_pct": int,   # 信息完整度 0-100
            "phase": str,            # "collecting" | "completed"
            "assessment": str|None   # 诊断报告（completed 时才有值）
        }
    """
    graph = get_doctor_graph()
    config = {"configurable": {"thread_id": session_id}}

    # 已完成的会话不再继续（避免误操作覆盖诊断）
    try:
        current = graph.get_state(config)
        if current.values and current.values.get("collection_phase") == "completed":
            return {
                "reply": "本次问诊已完成，诊断报告已生成。如需重新问诊，请使用新的会话 ID。",
                "phase": "completed",
                "assessment": current.values.get("assessment"),
            }
    except Exception:
        pass

    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )

    # 取最后一条 AIMessage 作为医生回复
    reply = ""
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage):
            reply = m.content
            break

    return {
        "reply": reply,
        "phase": result.get("collection_phase", "collecting"),
        "assessment": result.get("assessment"),
    }


def get_session_info(session_id: str) -> dict:
    """
    获取指定会话的当前问诊状态摘要。

    Returns:
        包含 session_id、patient_info、phase、turn_count、completion_pct、assessment 的字典，
        或 {"error": "..."} 若会话不存在。
    """
    graph = get_doctor_graph()
    config = {"configurable": {"thread_id": session_id}}

    try:
        state = graph.get_state(config)
    except Exception as e:
        return {"error": str(e)}

    if not state.values:
        return {"error": "会话不存在或尚未开始"}

    patient_info = state.values.get("patient_info") or {}
    return {
        "session_id": session_id,
        "patient_info": patient_info,
        "phase": state.values.get("collection_phase", "collecting"),
        "turn_count": state.values.get("turn_count", 0),
        "completion_pct": _calc_completion_pct(patient_info),
        "assessment": state.values.get("assessment"),
    }


# ---------------------------------------------------------------------------
# HTTP API views
# ---------------------------------------------------------------------------


async def doctor_chat_api(request: Request):
    """
    POST /ai/doctor/chat

    Request body:
        {
            "session_id": "可选，不传则自动生成",
            "message": "患者消息"
        }

    Response:
        {
            "code": 0,
            "msg": "ok",
            "data": {
                "session_id": "xxx",
                "reply": "医生回复",
                "phase": "collecting",
                "assessment": null
            }
        }
    """
    body = (await read_json_optional(request)) or {}
    session_id = (body.get("session_id") or "").strip() or str(uuid.uuid4())
    message = (body.get("message") or "").strip()

    if not message:
        return ({"code": 400, "msg": "message 不能为空"}, 400)

    result = chat(session_id, message)
    return {
        "code": 0,
        "msg": "ok",
        "data": {"session_id": session_id, **result},
    }


async def doctor_session_api(request: Request, session_id: str):
    """
    GET /ai/doctor/session/<session_id>

    返回当前会话的问诊状态摘要（不触发新一轮对话）。
    """
    result = get_session_info(session_id)
    if "error" in result:
        return ({"code": 404, "msg": result["error"]}, 404)
    return {"code": 0, "msg": "ok", "data": result}
