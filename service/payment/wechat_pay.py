"""
静态微信收款码 + 手动确认方案

流程：
1. 前端调 /api/payment/create → 拿到 out_trade_no + 展示二维码
2. 用户扫码付款后点「我已付款」→ 调 /api/payment/claim
3. 管理员在手机收到微信到账通知 → 调 /api/payment/admin/confirm 确认
4. 前端轮询 /api/payment/status，status=2 时放行下载

订单状态：
  0 - 待支付
  1 - 用户已申报（等待管理员确认）
  2 - 已确认，可下载
  3 - 已拒绝/关闭

环境变量：
  WECHAT_QR_PATH   - 微信收款二维码图片本地路径（如 static/payment/wechat_qr.png）
  PAYMENT_ADMIN_TOKEN - 管理员确认接口的鉴权 token（随机字符串即可）
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

WECHAT_QR_PATH = os.getenv("WECHAT_QR_PATH", "static/payment/wechat_qr.png")
PAYMENT_ADMIN_TOKEN = os.getenv("PAYMENT_ADMIN_TOKEN", "")

# PPT 下载定价（元）
PPT_DOWNLOAD_PRICE = float(os.getenv("PPT_DOWNLOAD_PRICE", "1.5"))


def new_order_no() -> str:
    return uuid.uuid4().hex


def get_qr_path() -> str:
    return WECHAT_QR_PATH


def verify_admin_token(token: str) -> bool:
    if not PAYMENT_ADMIN_TOKEN:
        raise ValueError("PAYMENT_ADMIN_TOKEN 未配置，请在 .env 中设置")
    return token == PAYMENT_ADMIN_TOKEN
