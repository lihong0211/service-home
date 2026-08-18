from sqlalchemy import Column, String, Text, Integer, Boolean, Index
from app.app import db
from model.common.base_model import BaseModel


class AgentTrace(db.Model, BaseModel):
    """
    LangGraph 图执行记录：一次图运行（invoke/stream 一轮）对应一条。
    身兼两个模块：【可观测性】——status/duration_ms/steps_detail 支撑排查"哪一步慢/哪一步错"；
    【回流机制】——feedback/feedback_note/copied 是回流的采集入口，status='error' 或 feedback='bad'
    的行就是 bad case 候选池，供 list_bad_cases() 查询（见 service/ai/langchain.py）；
    feedback（显式反馈）和 copied（隐式反馈：答案被复制过，往往比主动点踩更真实——
    大部分不满意用户是沉默流失，不会点踩）是好 case 也一样能有的信号，与 bad case 无关。
    """

    __bind_key__ = "ai"
    __tablename__ = "agent_trace"

    graph_name = Column(String(50), nullable=False, comment="图名称: router/loop/parallel/hitl")
    thread_id = Column(String(100), nullable=True, comment="会话线程ID")
    user_id = Column(String(64), nullable=True, comment="用户ID")
    input_summary = Column(Text, nullable=True, comment="输入摘要(query/input_text)")
    output_summary = Column(Text, nullable=True, comment="输出摘要(response)")
    status = Column(String(20), nullable=False, default="success", comment="success/error")
    error_message = Column(Text, nullable=True, comment="失败原因")
    total_steps = Column(Integer, nullable=True, comment="节点执行步数")
    duration_ms = Column(Integer, nullable=True, comment="总耗时(ms)")
    steps_detail = Column(Text, nullable=True, comment="每步 nodeId/duration_ms 的 JSON，供排查问题时展开")
    # 【回流机制】显式反馈：用户/人工对这次回答的评价，null=未评价，是 bad case 挖掘的信号源之一
    feedback = Column(String(10), nullable=True, comment="用户反馈: good/bad，null=未反馈")
    feedback_note = Column(Text, nullable=True, comment="反馈备注，如具体哪里答错了")
    # 【回流机制】隐式反馈：答案是否被用户复制过。不像点赞点踩需要用户主动操作，复制是
    # 用户已经在使用答案的自然动作，信号更真实
    copied = Column(Boolean, nullable=False, default=False, comment="隐式反馈: 答案是否被复制过")

    __table_args__ = (
        Index("ix_agent_trace_graph_name", "graph_name"),
        Index("ix_agent_trace_thread_id", "thread_id"),
        Index("ix_agent_trace_user_id", "user_id"),
        Index("ix_agent_trace_feedback", "feedback"),
    )
