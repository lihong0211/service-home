# service/peach/check_result.py
"""
检查服务模块 -
"""
from flask import request, jsonify
from app.app import db
from model.peach import CheckResult


def add():
    """增加检查记录"""
    data = request.get_json()
    platform = data.get("platform")
    patientSex = data.get("patientSex")
    patientAge = data.get("patientAge")
    primaryDiagnosis = data.get("primaryDiagnosis")
    medicines = data.get("medicines")
    pass_flag = data.get("pass")
    params = data.get("params")
    error = data.get("error")
    isNotMatch = data.get("isNotMatch")
    medicineName = data.get("fullName")
    specification = data.get("specification")
    takeDirection = data.get("takeDirection")
    takeFrequence = data.get("takeFrequence")
    medicineAmount = data.get("medicineAmount")
    takeDose = data.get("takeDose")
    formType = data.get("formType")

    if not all(
        [medicineName, specification, takeDirection, takeFrequence, takeDose, formType]
    ):
        return jsonify({"code": 500, "msg": "缺少参数"})

    try:
        check_data = {
            "platform": platform,
            "patientSex": patientSex,
            "patientAge": patientAge,
            "primaryDiagnosis": primaryDiagnosis,
            "medicines": medicines,
            "pass_flag": pass_flag,
            "params": params,
            "error": error,
            "isNotMatch": isNotMatch,
            "medicineName": medicineName,
            "specification": specification,
            "takeDirection": takeDirection,
            "takeFrequence": takeFrequence,
            "medicineAmount": medicineAmount,
            "takeDose": takeDose,
            "formType": formType,
        }
        CheckResult.insert(check_data)

        return jsonify({"code": 200, "msg": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 500, "msg": str(e)})


def list():
    """查询检查记录列表"""
    data = request.get_json() if request.is_json else {}
    page = data.get("page", 1)
    size = min(data.get("size", 10), 10000)  # 默认10条，最大50条
    query = data.get("query")

    try:
        # 构建查询条件
        criterion = {}
        if query:
            for key, value in query.items():
                if value:
                    criterion[key] = {"type": "like", "value": value}

        # 获取总数
        # total = CheckResult.count(criterion)

        # 获取分页数据
        offset = (page - 1) * size
        # 只查询实际存在的字段，避免查询不存在的 create_at、update_at、deleted_at、pdd_report
        results = (
            CheckResult.builder_query(criterion)
            .with_entities(
                CheckResult.id,
                CheckResult.platform,
                CheckResult.patientSex,
                CheckResult.patientAge,
                CheckResult.primaryDiagnosis,
                CheckResult.medicines,
                CheckResult.pass_flag,
                CheckResult.params,
                CheckResult.error,
                CheckResult.isNotMatch,
                CheckResult.medicineName,
                CheckResult.specification,
                CheckResult.takeDirection,
                CheckResult.takeFrequence,
                CheckResult.medicineAmount,
                CheckResult.takeDose,
                CheckResult.formType,
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
                    "platform": item[1],
                    "patientSex": item[2],
                    "patientAge": item[3],
                    "primaryDiagnosis": item[4],
                    "medicines": item[5],
                    "pass_flag": item[6],
                    "params": item[7],
                    "error": item[8],
                    "isNotMatch": item[9],
                    "medicineName": item[10],
                    "specification": item[11],
                    "takeDirection": item[12],
                    "takeFrequence": item[13],
                    "medicineAmount": item[14],
                    "takeDose": item[15],
                    "formType": item[16],
                }
            )

        return jsonify(
            {
                "code": 200,
                "data": {
                    "data": data_list,
                    "total": 5018980,
                    "page": page,
                },
            }
        )
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)})
