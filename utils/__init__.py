# utils/__init__.py
"""
工具函数模块
"""

def extend(target, source, flag=None):
    """
    扩展对象，将source的属性复制到target
    """
    for key in source:
        if hasattr(source, key) or key in source:
            if flag:
                target[key] = source[key]
            else:
                if key not in target or target[key] is None:
                    target[key] = source[key]
    return target


def limit(table, page_num, page_size, query=None):
    """
    生成分页查询SQL
    :param table: 表名
    :param page_num: 分页页数
    :param page_size: 分页条数
    :param query: 查询对象 例：{'id': 1, 'name': '小明'}
    :returns: sql语句
    """
    sql = "WHERE "
    if query:
        key_list = list(query.keys())
        # 查询对象全为空则结束函数
        if not any(query.get(e) for e in key_list):
            sql = ''
        else:
            # 生成SQL语句
            conditions = []
            for e in key_list:
                if query.get(e):
                    conditions.append(f"{e} LIKE '%{query[e]}%'")
            if conditions:
                sql = "WHERE " + " AND ".join(conditions) + " "
            else:
                sql = ''
    else:
        sql = ''
    
    offset = (page_num - 1) * page_size
    result_sql = f"SELECT * FROM {table} {sql}LIMIT {offset},{page_size}"
    return result_sql


def try_json_parse(data):
    """
    尝试解析JSON字符串
    """
    import json
    if data is None:
        return []
    if isinstance(data, str):
        try:
            return json.loads(data)
        except:
            return []
    return data


