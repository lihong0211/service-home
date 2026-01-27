# service/words/__init__.py
"""
单词服务模块 - 使用ORM
"""
import json
from flask import request, jsonify
from app.app import app, db
from model.words_model import Words
from utils import try_json_parse


def add():
    """增加单词"""
    data = request.get_json()
    word = data.get('word')
    word_type = data.get('type')
    meaning = data.get('meaning')
    root = data.get('root')
    root_case = data.get('root_case')
    affix = data.get('affix')
    affix_case = data.get('affix_case')
    collocation = data.get('collocation')
    collocation_meaning = data.get('collocation_meaning')
    sentence = data.get('sentence')
    mastered = data.get('mastered', 0)
    
    if not word or not meaning:
        return jsonify({
            'code': 500,
            'msg': 'word or meaning is empty',
        })
    
    try:
        # 检查单词是否已存在
        existing = Words.query.filter(Words.word == word, Words.deleted_at.is_(None)).first()
        if existing:
            return jsonify({
                'code': 500,
                'msg': '单词已存在',
            })
        
        # 插入新单词
        word_type_str = ','.join(word_type) if isinstance(word_type, list) else word_type
        word_data = {
            'word': word,
            'type': word_type_str,
            'meaning': meaning,
            'root': root,
            'root_case': root_case,
            'affix': affix,
            'affix_case': affix_case,
            'collocation': collocation,
            'collocation_meaning': collocation_meaning,
            'sentence': sentence,
            'mastered': mastered,
        }
        Words.insert(word_data)
        
        return jsonify({
            'code': 200,
            'msg': 'success',
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'code': 500,
            'msg': str(e),
        })


def delete():
    """删除单词"""
    data = request.get_json()
    word_id = data.get('id')
    
    try:
        Words.delete(word_id)
        return jsonify({
            'code': 200,
            'msg': 'success',
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'code': 500,
            'msg': str(e),
        })


def update():
    """更新单词"""
    data = request.get_json()
    word = data.get('word')
    word_type = data.get('type')
    meaning = data.get('meaning')
    root = data.get('root')
    root_case = data.get('root_case')
    affix = data.get('affix')
    affix_case = data.get('affix_case')
    collocation = data.get('collocation')
    collocation_meaning = data.get('collocation_meaning')
    sentence = data.get('sentence')
    mastered = data.get('mastered')
    word_id = data.get('id')
    
    try:
        word_type_str = ','.join(word_type) if isinstance(word_type, list) else word_type
        word_data = {
            'id': word_id,
            'word': word,
            'type': word_type_str,
            'meaning': meaning,
            'root': root,
            'root_case': root_case,
            'affix': affix,
            'affix_case': affix_case,
            'collocation': collocation,
            'collocation_meaning': collocation_meaning,
            'sentence': sentence,
            'mastered': mastered,
        }
        Words.update(word_data)
        
        return jsonify({
            'code': 200,
            'msg': 'success',
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'code': 500,
            'msg': str(e),
        })


def list_words():
    """查询单词列表"""
    data = request.get_json() if request.is_json else {}
    page = data.get('page', 1)
    size = data.get('size', 10)
    query = data.get('query')
    
    try:
        # 构建查询条件
        criterion = {}
        if query:
            for key, value in query.items():
                if value:
                    criterion[key] = {'type': 'like', 'value': value}
        
        # 获取总数
        total = Words.count(criterion)
        
        # 获取分页数据
        offset = (page - 1) * size
        # 只查询实际存在的字段，避免查询不存在的 create_at、update_at、deleted_at
        words_list = (
            Words.builder_query(criterion)
            .with_entities(
                Words.id,
                Words.word,
                Words.type,
                Words.meaning,
                Words.root,
                Words.root_case,
                Words.affix,
                Words.affix_case,
                Words.collocation,
                Words.collocation_meaning,
                Words.sentence,
                Words.mastered
            )
            .offset(offset)
            .limit(size)
            .all()
        )
        
        # 处理数据（with_entities 返回元组）
        data_list = []
        for item in words_list:
            data_list.append({
                'id': item[0],
                'word': item[1],
                'type': item[2].split(',') if item[2] else [],
                'meaning': item[3],
                'root': item[4],
                'root_case': item[5],
                'affix': item[6],
                'affix_case': item[7],
                'collocation': item[8],
                'collocation_meaning': item[9],
                'sentence': item[10],
                'mastered': item[11] or 0 if item[11] is not None else 0,
            })
        
        return jsonify({
            'code': 200,
            'data': {
                'data': data_list,
                'total': total,
                'page': page,
            },
        })
    except Exception as e:
        return jsonify({
            'code': 500,
            'msg': str(e),
        })
