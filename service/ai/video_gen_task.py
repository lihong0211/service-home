"""
火山方舟文生视频任务：落库与按 task_id 查询，支持从 Ark API 拉取最新状态。
"""
import os
from typing import Optional

from app.app import db
from model.ai.video_gen_task import VideoGenTask


# 火山方舟查询任务 API（仅支持最近 7 天）
ARK_TASK_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"


def _ark_api_key() -> Optional[str]:
    """从环境变量读取火山方舟 API Key，用于拉取任务状态。"""
    return os.environ.get("VOLCANO_ARK_API_KEY") or os.environ.get("ARK_API_KEY")


def create_task(
    task_id: str,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    resolution: Optional[str] = None,
    ratio: Optional[str] = None,
    duration: Optional[int] = None,
    source: Optional[str] = None,
) -> dict:
    """
    创建一条文生视频任务记录（如 Dify 提交任务后调用）。
    若 task_id 已存在则更新部分字段并返回。
    """
    row = VideoGenTask.query.filter_by(task_id=task_id).first()
    if row:
        if prompt is not None:
            row.prompt = prompt
        if model is not None:
            row.model = model
        if resolution is not None:
            row.resolution = resolution
        if ratio is not None:
            row.ratio = ratio
        if duration is not None:
            row.duration = duration
        if source is not None:
            row.source = source
        db.session.commit()
        return _row_to_dict(row)

    row = VideoGenTask(
        task_id=task_id,
        status="submitted",
        prompt=prompt,
        model=model,
        resolution=resolution,
        ratio=ratio,
        duration=duration,
        source=source,
    )
    db.session.add(row)
    db.session.commit()
    return _row_to_dict(row)


def _row_to_dict(row: VideoGenTask) -> dict:
    return {
        "id": row.id,
        "task_id": row.task_id,
        "status": row.status,
        "prompt": row.prompt,
        "model": row.model,
        "video_url": row.video_url,
        "resolution": row.resolution,
        "ratio": row.ratio,
        "duration": row.duration,
        "source": row.source,
        "create_at": row.create_at.isoformat() if row.create_at else None,
        "update_at": row.update_at.isoformat() if row.update_at else None,
    }


def get_task(task_id: str, sync_from_ark: bool = False) -> Optional[dict]:
    """
    按 task_id 查询任务。若 sync_from_ark=True 且配置了 API Key，会请求火山方舟接口并更新本地状态后返回。
    """
    row = VideoGenTask.query.filter_by(task_id=task_id).first()
    if sync_from_ark and _ark_api_key():
        _fetch_and_update_from_ark(task_id)
        row = VideoGenTask.query.filter_by(task_id=task_id).first()
    if not row:
        return None
    return _row_to_dict(row)


def _fetch_and_update_from_ark(task_id: str) -> None:
    """请求火山方舟「查询视频生成任务信息」接口，并更新本地记录。"""
    import requests

    url = f"{ARK_TASK_URL}/{task_id}"
    headers = {"Authorization": f"Bearer {_ark_api_key()}"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return

    row = VideoGenTask.query.filter_by(task_id=task_id).first()
    status = (data.get("status") or "").strip().lower()
    video_url = None
    content = data.get("content") or {}
    if isinstance(content, dict):
        video_url = content.get("video_url")
    elif isinstance(content, str):
        video_url = content

    if row:
        row.status = status if status else row.status
        if video_url:
            row.video_url = video_url
        if data.get("model"):
            row.model = data.get("model")
        if data.get("resolution"):
            row.resolution = data.get("resolution")
        if data.get("ratio"):
            row.ratio = data.get("ratio")
        if data.get("duration") is not None:
            row.duration = data.get("duration")
        db.session.commit()
    else:
        # 远端有记录但本地没有，则插入一条
        row = VideoGenTask(
            task_id=task_id,
            status=status or "unknown",
            prompt=None,
            model=data.get("model"),
            video_url=video_url,
            resolution=data.get("resolution"),
            ratio=data.get("ratio"),
            duration=data.get("duration"),
            source=None,
        )
        db.session.add(row)
        db.session.commit()


def list_tasks(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
) -> dict:
    """分页查询任务列表，可按 status 筛选。"""
    q = VideoGenTask.query.filter(VideoGenTask.deleted_at.is_(None)).order_by(VideoGenTask.create_at.desc())
    if status:
        q = q.filter(VideoGenTask.status == status.strip().lower())
    total = q.count()
    items = q.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [_row_to_dict(r) for r in items],
    }


# ---------- Flask API 入口 ----------


def video_gen_task_create_api():
    """
    POST /ai/video-gen/tasks
    Body: { "task_id": "cgt-xxx", "prompt": "...", "model": "...", "resolution": "720p", "ratio": "16:9", "duration": 5, "source": "dify" }
    Dify 在提交文生视频任务后可调用此接口，将 task_id 写入数据库，便于后续查询。
    """
    from flask import request, jsonify

    data = request.get_json() or {}
    task_id = (data.get("task_id") or "").strip()
    if not task_id:
        return jsonify({"code": 400, "msg": "缺少 task_id"}), 400
    try:
        out = create_task(
            task_id=task_id,
            prompt=data.get("prompt"),
            model=data.get("model"),
            resolution=data.get("resolution"),
            ratio=data.get("ratio"),
            duration=data.get("duration"),
            source=data.get("source"),
        )
        return jsonify({"code": 0, "msg": "ok", "data": out})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


def video_gen_task_get_api(task_id: str):
    """
    GET /ai/video-gen/tasks/<task_id>?sync=1
    sync=1 时从火山方舟拉取最新状态并更新本地后返回（需配置 VOLCANO_ARK_API_KEY 或 ARK_API_KEY）。
    """
    from flask import request, jsonify

    task_id = (task_id or "").strip()
    if not task_id:
        return jsonify({"code": 400, "msg": "缺少 task_id"}), 400
    sync = request.args.get("sync", "").lower() in ("1", "true", "yes")
    out = get_task(task_id, sync_from_ark=sync)
    if out is None:
        return jsonify({"code": 404, "msg": "任务不存在"}), 404
    return jsonify({"code": 0, "msg": "ok", "data": out})


def video_gen_task_list_api():
    """
    GET /ai/video-gen/tasks?page=1&page_size=20&status=succeeded
    """
    from flask import request, jsonify

    page = max(1, int(request.args.get("page", 1)))
    page_size = min(100, max(1, int(request.args.get("page_size", 20))))
    status = request.args.get("status", "").strip() or None
    out = list_tasks(page=page, page_size=page_size, status=status)
    return jsonify({"code": 0, "msg": "ok", "data": out})
