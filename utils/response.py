"""通用 API 响应辅助。"""
from flask import jsonify


def api_response(out: dict, *, error_code: int = 400, error_code_fn=None, error_data=None, success_data=None):
    """
    统一处理 run() 返回的 dict（含 error 键）。
    :param out: run 函数返回的 dict，成功时 error=None
    :param error_code: 出错时的 HTTP 状态码
    :param error_code_fn: 可选，callable(out) -> int 动态计算错误码
    :param error_data: 出错时 data 字段，默认用 out
    :param success_data: 成功时 data 字段，默认 out 去掉 error
    :return: (Response, status_code)
    """
    if out.get("error"):
        code = error_code_fn(out) if error_code_fn else error_code
        data = out if error_data is None else error_data
        return jsonify({"code": code, "msg": out["error"], "data": data}), code
    data = {k: v for k, v in out.items() if k != "error"} if success_data is None else success_data
    return jsonify({"code": 0, "msg": "ok", "data": data}), 200
