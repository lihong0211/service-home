# routes/english.py
"""英语相关路由：单词、词根、词缀、对话、生活用语。"""

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


def register_english(bp):
    bp.add_url_rule("/english/words/add", "words_add", words_add, methods=["POST"])
    bp.add_url_rule("/english/words/delete", "words_delete", words_delete, methods=["POST"])
    bp.add_url_rule("/english/words/update", "words_update", words_update, methods=["POST"])
    bp.add_url_rule("/english/words/list", "words_list", list_words, methods=["POST"])

    bp.add_url_rule("/english/root/add", "root_add", root_add, methods=["POST"])
    bp.add_url_rule("/english/root/delete", "root_delete", root_delete, methods=["POST"])
    bp.add_url_rule("/english/root/update", "root_update", root_update, methods=["POST"])
    bp.add_url_rule("/english/root/list", "root_list", list_roots, methods=["POST"])

    bp.add_url_rule("/english/affix/add", "affix_add", affix_add, methods=["POST"])
    bp.add_url_rule("/english/affix/delete", "affix_delete", affix_delete, methods=["POST"])
    bp.add_url_rule("/english/affix/update", "affix_update", affix_update, methods=["POST"])
    bp.add_url_rule("/english/affix/list", "affix_list", list_affixes, methods=["GET"])

    bp.add_url_rule("/english/dialogue/add", "dialogue_add", dialogue_add, methods=["POST"])
    bp.add_url_rule("/english/dialogue/delete", "dialogue_delete", dialogue_delete, methods=["POST"])
    bp.add_url_rule("/english/dialogue/update", "dialogue_update", dialogue_update, methods=["POST"])
    bp.add_url_rule("/english/dialogue/list", "dialogue_list", list_dialogues, methods=["GET"])

    bp.add_url_rule("/english/living-speech/add", "living_speech_add", living_speech_add, methods=["POST"])
    bp.add_url_rule("/english/living-speech/delete", "living_speech_delete", living_speech_delete, methods=["POST"])
    bp.add_url_rule("/english/living-speech/update", "living_speech_update", living_speech_update, methods=["POST"])
    bp.add_url_rule("/english/living-speech/list", "living_speech_list", list_speeches, methods=["POST"])
