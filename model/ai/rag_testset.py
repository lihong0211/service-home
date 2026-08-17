"""
RAG 测试集表——RAGAS 自动生成的「问题-上下文-标准答案」三元组，落库供反复做回归评测用
（不是生成一次看一眼就扔），见 service/ai/rag_eval.py 的 generate_testset_api。
"""
from sqlalchemy import Column, String, Text, Integer
from app.app import db
from model.common.base_model import BaseModel


class RagTestsetItem(db.Model, BaseModel):
    __tablename__ = "rag_testset_item"
    __bind_key__ = "ai"

    kb_id = Column(Integer, nullable=False, index=True, comment="所属知识库 id")
    kb_name = Column(String(255), nullable=True, comment="所属知识库名称（冗余，方便展示）")
    question = Column(Text, nullable=False, comment="RAGAS 生成的问题")
    ground_truth = Column(Text, nullable=False, comment="RAGAS 生成的标准答案（模型合成，非人工核对）")
    source_context = Column(Text, nullable=True, comment="生成时依据的原文片段，供人工核对标准答案是否靠谱")
    synthesizer = Column(String(100), nullable=True, comment="生成方式：single_hop_specific/multi_hop_abstract 等")
