"""
支付订单模型
"""
from sqlalchemy import Column, String, Numeric, SmallInteger, Text
from sqlalchemy.dialects.mysql import INTEGER
from app.app import db
from model.common.base_model import BaseModel


class PayOrder(db.Model, BaseModel):
    """
    支付订单表 pay_order
    status: 0=待支付  1=已支付  2=已关闭
    biz_type: 枚举业务来源，当前只有 ppt_download
    """
    __tablename__ = "pay_order"
    __bind_key__ = "ai"

    id = Column(INTEGER(11), primary_key=True, autoincrement=True)
    out_trade_no = Column(String(64), nullable=False, unique=True, comment="商户订单号")
    trade_no = Column(String(64), nullable=True, comment="第三方平台流水号")
    biz_type = Column(String(32), nullable=False, default="ppt_download", comment="业务类型")
    biz_id = Column(String(128), nullable=True, comment="业务 ID，如 ppt_id")
    subject = Column(String(128), nullable=False, comment="商品名称")
    amount = Column(Numeric(10, 2), nullable=False, comment="金额（元）")
    status = Column(SmallInteger, nullable=False, default=0, comment="0待付 1已付 2关闭")
    pay_type = Column(String(16), nullable=True, comment="wxpay / alipay")
    extra = Column(Text, nullable=True, comment="扩展 JSON，如下载凭证")
