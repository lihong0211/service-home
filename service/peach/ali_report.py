# service/peach/ali_report.py
"""
阿里报告服务模块 -
"""
from flask import request, jsonify
from app.app import db
from model.peach import AliRpCheck


def add():
    """增加阿里报告"""
    data = request.get_json()
    pharmacist = data.get("pharmacist")
    patientSex = data.get("patientSex")
    patientAge = data.get("patientAge")
    primaryDiagnosis = data.get("primaryDiagnosis")
    medicines = data.get("medicines")
    pass_flag = data.get("pass")
    reason = data.get("reason")
    query = data.get("query")
    rpID = data.get("rpID")

    try:
        ali_data = {
            "pharmacist": pharmacist,
            "patientSex": patientSex,
            "patientAge": patientAge,
            "primaryDiagnosis": primaryDiagnosis,
            "medicines": medicines,
            "pass": pass_flag,  # pass字段会在BaseModel.loads中自动映射为pass_flag
            "reason": reason,
            "query": query,
            "rpID": rpID,
            "refuse": 0,
        }
        AliRpCheck.insert(ali_data)

        return jsonify({"code": 200, "msg": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "msg": str(e)})


def get():
    """获取阿里报告"""
    data = request.get_json() if request.is_json else {}
    rpID = data.get("rpID")

    if not rpID:
        return jsonify({"code": 400, "msg": "Missing rpID parameter"})

    try:
        result = AliRpCheck.select_one_by({"rpID": rpID})

        if result:
            return jsonify(
                {
                    "code": 200,
                    "msg": "success",
                    "data": {"refuse": result.refuse or 0},
                }
            )
        else:
            return jsonify({"code": 200, "msg": "success", "data": {"refuse": 0}})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})


def update():
    """更新阿里报告"""
    data = request.get_json()
    rpID = data.get("rpID")
    costTime = data.get("costTime")

    try:
        if costTime:
            # 更新耗时（只更新更小的值或NULL）
            result = AliRpCheck.select_one_by({"rpID": rpID})
            if result:
                if result.costTime is None or result.costTime > costTime:
                    AliRpCheck.update({"id": result.id, "costTime": costTime})
        else:
            # 增加拒绝次数 - 使用原生SQL更新，因为需要原子操作
            from sqlalchemy import text

            db.session.execute(
                text("UPDATE ali_rp_check SET refuse = refuse + 1 WHERE rpID = :rpID"),
                {"rpID": rpID},
            )
            db.session.commit()

        return jsonify({"code": 200, "msg": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "msg": str(e)})


def list():
    """查询阿里报告列表"""
    data = request.get_json() if request.is_json else {}
    page = data.get("page", 1)
    size = data.get("size", 10)
    query = data.get("query")

    try:
        # 构建查询条件
        criterion = {}
        if query:
            for key, value in query.items():
                if value:
                    criterion[key] = {"type": "like", "value": value}

        # 获取总数
        # total = AliRpCheck.count(criterion)

        # 获取分页数据
        offset = (page - 1) * size
        # 只查询实际存在的字段，避免查询不存在的 create_at、update_at、deleted_at
        results = (
            AliRpCheck.builder_query(criterion)
            .with_entities(
                AliRpCheck.id,
                AliRpCheck.pharmacist,
                AliRpCheck.patientSex,
                AliRpCheck.patientAge,
                AliRpCheck.primaryDiagnosis,
                AliRpCheck.medicines,
                AliRpCheck.pass_flag,
                AliRpCheck.reason,
                AliRpCheck.query,
                AliRpCheck.rpID,
                AliRpCheck.refuse,
                AliRpCheck.costTime,
            )
            .offset(offset)
            .limit(size)
            .all()
        )

        data_list = []
        for item in results:
            data_list.append(
                {
                    "id": item[0],
                    "pharmacist": item[1],
                    "patientSex": item[2],
                    "patientAge": item[3],
                    "primaryDiagnosis": item[4],
                    "medicines": item[5],
                    "pass_flag": item[6],
                    "reason": item[7],
                    "query": item[8],
                    "rpID": item[9],
                    "refuse": item[10] or 0,
                    "costTime": item[11],
                }
            )

        return jsonify(
            {
                "code": 200,
                "data": {
                    "data": data_list,
                    "total": 1294953,
                    "page": page,
                },
            }
        )
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})
