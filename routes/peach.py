# routes/peach.py
"""Peach 相关路由：阿里报告、检查结果、插件统计。"""

from service.peach.ali_report import (
    add as ali_report_add,
    get as ali_report_get,
    update as ali_report_update,
    list as ali_report_list,
)
from service.peach.check_result import (
    add as check_result_add,
    list as check_result_list,
)
from service.peach.plugin_statistic import (
    add as plugin_statistics_add,
    list_statistics,
    detail as plugin_statistics_detail,
)


def register_peach(bp):
    bp.add_url_rule("/peach/ali-report/add", "ali_report_add", ali_report_add, methods=["POST"])
    bp.add_url_rule("/peach/ali-report/get", "ali_report_get", ali_report_get, methods=["POST"])
    bp.add_url_rule("/peach/ali-report/update", "ali_report_update", ali_report_update, methods=["POST"])
    bp.add_url_rule("/peach/ali-report/list", "ali_report_list", ali_report_list, methods=["POST"])

    bp.add_url_rule("/peach/check-result/add", "check_result_add", check_result_add, methods=["POST"])
    bp.add_url_rule("/peach/check-result/list", "check_result_list", check_result_list, methods=["POST"])

    bp.add_url_rule("/peach/plugin-statistics/add", "plugin_statistics_add", plugin_statistics_add, methods=["POST"])
    bp.add_url_rule("/peach/plugin-statistics/list", "plugin_statistics_list", list_statistics, methods=["POST"])
    bp.add_url_rule("/peach/plugin-statistics/detail", "plugin_statistics_detail", plugin_statistics_detail, methods=["POST"])
