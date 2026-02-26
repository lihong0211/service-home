"""
彩虹易支付（epay）对接模块
文档：https://www.epay.pro/doc/
个人微信收款：在 epay 后台绑定个人微信收款码即可，无需企业微信支付。

环境变量：
  EPAY_API_URL   - 你的 epay 站点，如 https://pay.example.com
  EPAY_PID       - 商户 ID
  EPAY_KEY       - 商户密钥（MD5 签名用）
  EPAY_NOTIFY_URL - 异步回调地址（公网可访问），如 https://yourserver.com/api/payment/notify
  EPAY_RETURN_URL - 同步跳转地址，如 https://yourserver.com/ppt
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional
from urllib.parse import urlencode

EPAY_API_URL = os.getenv("EPAY_API_URL", "").rstrip("/")
EPAY_PID = os.getenv("EPAY_PID", "")
EPAY_KEY = os.getenv("EPAY_KEY", "")
EPAY_NOTIFY_URL = os.getenv("EPAY_NOTIFY_URL", "")
EPAY_RETURN_URL = os.getenv("EPAY_RETURN_URL", "")


# ---------- 签名 ----------

def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _sign(params: dict) -> str:
    """
    epay 签名规则：
    1. 过滤 sign / sign_type / 空值
    2. 按 key 字典序排列
    3. 拼接 key=val&key=val
    4. 末尾拼接商户密钥
    5. MD5 小写
    """
    filtered = {k: v for k, v in params.items() if k not in ("sign", "sign_type") and v != ""}
    sorted_str = "&".join(f"{k}={v}" for k, v in sorted(filtered.items()))
    return _md5(sorted_str + EPAY_KEY)


def verify_sign(params: dict) -> bool:
    """验证 epay 回调签名是否合法。"""
    expected = _sign(params)
    return expected == params.get("sign", "")


# ---------- 生成支付链接 ----------

def build_pay_url(
    out_trade_no: str,
    amount: float,
    subject: str,
    pay_type: str = "wxpay",
    notify_url: Optional[str] = None,
    return_url: Optional[str] = None,
) -> str:
    """
    构造跳转到 epay 收银台的支付链接（GET 方式）。

    :param out_trade_no: 商户唯一订单号
    :param amount:       金额（元，保留2位小数）
    :param subject:      商品名称
    :param pay_type:     wxpay / alipay
    :param notify_url:   覆盖默认异步通知地址
    :param return_url:   覆盖默认同步跳转地址
    :return:             完整支付 URL
    """
    if not EPAY_API_URL or not EPAY_PID or not EPAY_KEY:
        raise ValueError("EPAY_API_URL / EPAY_PID / EPAY_KEY 未配置，请在 .env 中设置")

    params = {
        "pid": EPAY_PID,
        "type": pay_type,
        "out_trade_no": out_trade_no,
        "notify_url": notify_url or EPAY_NOTIFY_URL,
        "return_url": return_url or EPAY_RETURN_URL,
        "name": subject,
        "money": f"{amount:.2f}",
        "sign_type": "MD5",
    }
    params["sign"] = _sign(params)
    return f"{EPAY_API_URL}/submit.php?{urlencode(params)}"


# ---------- 解析回调 ----------

def parse_notify(form: dict) -> dict:
    """
    解析 epay 异步回调 POST 数据，验签并返回结构化结果。

    :param form: request.form（或等效 dict）
    :return: {
        "valid": bool,          # 签名是否合法
        "trade_no": str,        # epay 平台流水号
        "out_trade_no": str,    # 商户订单号
        "amount": float,        # 实付金额
        "trade_status": str,    # TRADE_SUCCESS / ...
        "paid": bool,           # 是否支付成功
    }
    """
    valid = verify_sign(dict(form))
    return {
        "valid": valid,
        "trade_no": form.get("trade_no", ""),
        "out_trade_no": form.get("out_trade_no", ""),
        "amount": float(form.get("money", 0)),
        "trade_status": form.get("trade_status", ""),
        "paid": valid and form.get("trade_status") == "TRADE_SUCCESS",
    }
