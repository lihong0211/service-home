"""
微信静态收款码支付路由

用户侧：
  POST /api/payment/create          - 创建订单
  GET  /api/payment/qrcode          - 获取微信收款二维码图片
  POST /api/payment/claim           - 用户申报「我已付款」
  GET  /api/payment/status          - 查询订单状态（前端轮询）

管理员侧（需 admin_token）：
  GET  /api/payment/admin/list      - 待确认订单列表
  POST /api/payment/admin/confirm   - 确认收款
  POST /api/payment/admin/reject    - 拒绝订单
"""
from __future__ import annotations

import os
from flask import send_file, request, jsonify

from app.app import db
from model.payment.pay_order import PayOrder
from service.payment.wechat_pay import (
    new_order_no,
    get_qr_path,
    verify_admin_token,
    PPT_DOWNLOAD_PRICE,
)


def register_payment(bp):
    bp.add_url_rule("/payment/create",         "payment_create",        payment_create_api,        methods=["POST"])
    bp.add_url_rule("/payment/qrcode",         "payment_qrcode",        payment_qrcode_api,        methods=["GET"])
    bp.add_url_rule("/payment/claim",          "payment_claim",         payment_claim_api,          methods=["POST"])
    bp.add_url_rule("/payment/status",         "payment_status",        payment_status_api,        methods=["GET"])
    bp.add_url_rule("/payment/admin/list",     "payment_admin_list",    payment_admin_list_api,    methods=["GET"])
    bp.add_url_rule("/payment/admin/confirm",  "payment_admin_confirm", payment_admin_confirm_api, methods=["POST"])
    bp.add_url_rule("/payment/admin/reject",   "payment_admin_reject",  payment_admin_reject_api,  methods=["POST"])


# ─────────────────────────── 用户侧 ───────────────────────────

def payment_create_api():
    """
    创建支付订单。

    Body JSON（均可选）:
      biz_type : 业务类型，默认 "ppt_download"
      biz_id   : 业务 ID，如 ppt_id
      subject  : 商品名称，默认 "PPT下载"
      amount   : 金额（元），默认取 PPT_DOWNLOAD_PRICE 环境变量

    返回:
      out_trade_no : 商户订单号
      amount       : 金额
      qrcode_url   : 二维码接口地址（前端直接 <img src="..."> 展示）
    """
    body = request.get_json(silent=True) or {}
    biz_type = body.get("biz_type", "ppt_download")
    biz_id   = body.get("biz_id", "")
    subject  = body.get("subject") or "PPT下载"
    amount   = float(body.get("amount") or PPT_DOWNLOAD_PRICE)

    if not biz_id:
        return jsonify({"code": 400, "msg": "请提供 biz_id"}), 400

    out_trade_no = new_order_no()

    order = PayOrder()
    order.out_trade_no = out_trade_no
    order.biz_type     = biz_type
    order.biz_id       = biz_id
    order.subject      = subject
    order.amount       = amount
    order.status       = 0
    order.pay_type     = "wxpay"
    db.session.add(order)
    db.session.commit()

    return jsonify({
        "code": 0,
        "msg": "ok",
        "data": {
            "out_trade_no": out_trade_no,
            "amount": amount,
            "qrcode_url": "/api/payment/qrcode",
        },
    })


def payment_qrcode_api():
    """返回微信收款二维码图片（直接作为 <img> src 使用）。"""
    qr_path = get_qr_path()
    if not os.path.exists(qr_path):
        return jsonify({"code": 404, "msg": f"二维码图片未找到，请将图片放到 {qr_path}"}), 404
    ext = os.path.splitext(qr_path)[1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext.lstrip("."), "image/png")
    return send_file(qr_path, mimetype=mime)


def payment_claim_api():
    """
    用户点击「我已付款」，将订单状态从 0 → 1（待管理员确认）。

    Body JSON:
      out_trade_no : 订单号
    """
    body = request.get_json(silent=True) or {}
    out_trade_no = body.get("out_trade_no")
    if not out_trade_no:
        return jsonify({"code": 400, "msg": "请提供 out_trade_no"}), 400

    order = PayOrder.select_one_by({"out_trade_no": out_trade_no})
    if not order:
        return jsonify({"code": 404, "msg": "订单不存在"}), 404
    if order.status != 0:
        status_desc = {1: "已申报，等待确认", 2: "已确认", 3: "已关闭"}
        return jsonify({"code": 400, "msg": status_desc.get(order.status, "订单状态异常")}), 400

    order.status = 1
    db.session.commit()

    return jsonify({"code": 0, "msg": "已提交，请等待确认（通常几分钟内）"})


def payment_status_api():
    """
    GET /api/payment/status?out_trade_no=xxx

    返回 status: 0=待支付 1=待确认 2=已确认可下载 3=已关闭
    """
    out_trade_no = request.args.get("out_trade_no")
    if not out_trade_no:
        return jsonify({"code": 400, "msg": "请提供 out_trade_no"}), 400

    order = PayOrder.select_one_by({"out_trade_no": out_trade_no})
    if not order:
        return jsonify({"code": 404, "msg": "订单不存在"}), 404

    status_desc = {0: "待支付", 1: "待确认", 2: "已确认，可下载", 3: "已关闭"}
    return jsonify({
        "code": 0,
        "msg": "ok",
        "data": {
            "out_trade_no": order.out_trade_no,
            "biz_id":       order.biz_id,
            "status":       order.status,
            "status_desc":  status_desc.get(order.status, "未知"),
            "amount":       float(order.amount),
            "paid":         order.status == 2,
        },
    })


# ─────────────────────────── 管理员侧 ───────────────────────────

def _check_admin(body: dict):
    """从请求 body 或 header 中取 admin_token 并校验。"""
    token = body.get("admin_token") or request.headers.get("X-Admin-Token", "")
    try:
        ok = verify_admin_token(token)
    except ValueError as e:
        return str(e)
    return None if ok else "admin_token 错误"


def payment_admin_list_api():
    """
    GET /api/payment/admin/list?admin_token=xxx&status=1
    列出指定状态的订单（默认列出 status=1 待确认）。
    """
    token = request.args.get("admin_token") or request.headers.get("X-Admin-Token", "")
    try:
        if not verify_admin_token(token):
            return jsonify({"code": 403, "msg": "admin_token 错误"}), 403
    except ValueError as e:
        return jsonify({"code": 500, "msg": str(e)}), 500

    status = int(request.args.get("status", 1))
    orders = PayOrder.select_by({"status": status, "order_by": {"col": "create_at", "sort": "desc"}})
    return jsonify({
        "code": 0,
        "msg": "ok",
        "data": [
            {
                "id":            o.id,
                "out_trade_no":  o.out_trade_no,
                "biz_id":        o.biz_id,
                "subject":       o.subject,
                "amount":        float(o.amount),
                "status":        o.status,
                "create_at":     o.create_at.strftime("%Y-%m-%d %H:%M:%S") if o.create_at else "",
            }
            for o in orders
        ],
    })


def payment_admin_confirm_api():
    """
    POST /api/payment/admin/confirm
    Body: { "out_trade_no": "xxx", "admin_token": "xxx" }
    确认收款，订单状态 1 → 2。
    """
    body = request.get_json(silent=True) or {}
    err = _check_admin(body)
    if err:
        return jsonify({"code": 403, "msg": err}), 403

    out_trade_no = body.get("out_trade_no")
    if not out_trade_no:
        return jsonify({"code": 400, "msg": "请提供 out_trade_no"}), 400

    order = PayOrder.select_one_by({"out_trade_no": out_trade_no})
    if not order:
        return jsonify({"code": 404, "msg": "订单不存在"}), 404
    if order.status not in (0, 1):
        return jsonify({"code": 400, "msg": f"订单当前状态 {order.status}，无法确认"}), 400

    order.status = 2
    db.session.commit()
    return jsonify({"code": 0, "msg": "已确认，用户可下载"})


def payment_admin_reject_api():
    """
    POST /api/payment/admin/reject
    Body: { "out_trade_no": "xxx", "admin_token": "xxx" }
    拒绝订单，状态 → 3。
    """
    body = request.get_json(silent=True) or {}
    err = _check_admin(body)
    if err:
        return jsonify({"code": 403, "msg": err}), 403

    out_trade_no = body.get("out_trade_no")
    if not out_trade_no:
        return jsonify({"code": 400, "msg": "请提供 out_trade_no"}), 400

    order = PayOrder.select_one_by({"out_trade_no": out_trade_no})
    if not order:
        return jsonify({"code": 404, "msg": "订单不存在"}), 404

    order.status = 3
    db.session.commit()
    return jsonify({"code": 0, "msg": "已拒绝"})
