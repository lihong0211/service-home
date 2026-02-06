# routes/__init__.py
"""
路由模块
"""
from flask import Blueprint

# 导入服务函数
from service.english.words import (
    add as words_add,
    delete as words_delete,
    update as words_update,
    list_words,
)
from service.english.root import (
    add as root_add,
    delete as root_delete,
    update as root_update,
    list_roots,
)
from service.english.affix import (
    add as affix_add,
    delete as affix_delete,
    update as affix_update,
    list_affixes,
)
from service.english.dialogue import (
    add as dialogue_add,
    delete as dialogue_delete,
    update as dialogue_update,
    list_dialogues,
)
from service.english.living_speech import (
    add as living_speech_add,
    delete as living_speech_delete,
    update as living_speech_update,
    list_speeches,
)
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

from service.ai.chat import chat as chat, ocr_chat as ocr_chat
from service.ai.stt import (
    transcribe as stt_transcribe,
    transcribe_stream as stt_transcribe_stream,
)
from service.ai.tts import speech as tts_speech
from service.ai.image_gen import generate as image_generate
from service.ai.video_undstanding import video_understand

# 创建蓝图
api_bp = Blueprint("api", __name__)

# 单词相关路由
api_bp.add_url_rule("/english/words/add", "words_add", words_add, methods=["POST"])
api_bp.add_url_rule(
    "/english/words/delete", "words_delete", words_delete, methods=["POST"]
)
api_bp.add_url_rule(
    "/english/words/update", "words_update", words_update, methods=["POST"]
)
api_bp.add_url_rule("/english/words/list", "words_list", list_words, methods=["POST"])

# 词根相关路由
api_bp.add_url_rule("/english/root/add", "root_add", root_add, methods=["POST"])
api_bp.add_url_rule(
    "/english/root/delete", "root_delete", root_delete, methods=["POST"]
)
api_bp.add_url_rule(
    "/english/root/update", "root_update", root_update, methods=["POST"]
)
api_bp.add_url_rule("/english/root/list", "root_list", list_roots, methods=["POST"])

# 词缀相关路由
api_bp.add_url_rule("/english/affix/add", "affix_add", affix_add, methods=["POST"])
api_bp.add_url_rule(
    "/english/affix/delete", "affix_delete", affix_delete, methods=["POST"]
)
api_bp.add_url_rule(
    "/english/affix/update", "affix_update", affix_update, methods=["POST"]
)
api_bp.add_url_rule("/english/affix/list", "affix_list", list_affixes, methods=["GET"])

# 对话相关路由
api_bp.add_url_rule(
    "/english/dialogue/add", "dialogue_add", dialogue_add, methods=["POST"]
)
api_bp.add_url_rule(
    "/english/dialogue/delete", "dialogue_delete", dialogue_delete, methods=["POST"]
)
api_bp.add_url_rule(
    "/english/dialogue/update", "dialogue_update", dialogue_update, methods=["POST"]
)
api_bp.add_url_rule(
    "/english/dialogue/list", "dialogue_list", list_dialogues, methods=["GET"]
)

# 生活用语相关路由
api_bp.add_url_rule(
    "/english/living-speech/add",
    "living_speech_add",
    living_speech_add,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/english/living-speech/delete",
    "living_speech_delete",
    living_speech_delete,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/english/living-speech/update",
    "living_speech_update",
    living_speech_update,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/english/living-speech/list", "living_speech_list", list_speeches, methods=["POST"]
)

# 阿里报告相关路由
api_bp.add_url_rule(
    "/peach/ali-report/add", "ali_report_add", ali_report_add, methods=["POST"]
)
api_bp.add_url_rule(
    "/peach/ali-report/get", "ali_report_get", ali_report_get, methods=["POST"]
)
api_bp.add_url_rule(
    "/peach/ali-report/update", "ali_report_update", ali_report_update, methods=["POST"]
)
api_bp.add_url_rule(
    "/peach/ali-report/list", "ali_report_list", ali_report_list, methods=["POST"]
)

# 检查结果相关路由
api_bp.add_url_rule(
    "/peach/check-result/add", "check_result_add", check_result_add, methods=["POST"]
)
api_bp.add_url_rule(
    "/peach/check-result/list", "check_result_list", check_result_list, methods=["POST"]
)

# 插件统计相关路由
api_bp.add_url_rule(
    "/peach/plugin-statistics/add",
    "plugin_statistics_add",
    plugin_statistics_add,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/peach/plugin-statistics/list",
    "plugin_statistics_list",
    list_statistics,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/peach/plugin-statistics/detail",
    "plugin_statistics_detail",
    plugin_statistics_detail,
    methods=["POST"],
)


api_bp.add_url_rule(
    "/ai/chat",
    "chat",
    chat,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/ai/orc",
    "ocr_chat",
    ocr_chat,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/ai/stt/transcribe",
    "stt_transcribe",
    stt_transcribe,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/ai/stt/transcribe_stream",
    "stt_transcribe_stream",
    stt_transcribe_stream,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/ai/tts",
    "tts_speech",
    tts_speech,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/ai/image/generate",
    "image_generate",
    image_generate,
    methods=["POST"],
)
api_bp.add_url_rule(
    "/ai/video/understand",
    "video_understand",
    video_understand,
    methods=["POST"],
)
