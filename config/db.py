# config/db.py
# MySQL数据库连接配置
import os
from typing import Dict, Any


def _get_db_config() -> Dict[str, Any]:
    """获取数据库配置（支持环境变量覆盖）"""
    config = {
        'host': os.environ.get('DB_HOST', 'localhost'),
        'user': os.environ.get('DB_USER', 'root'),
        'password': os.environ.get('DB_PASSWORD', 'Yy123456@'),
        'database': os.environ.get('DB_DATABASE', 'english'),
        'port': int(os.environ.get('DB_PORT', '3306')),
        'charset': os.environ.get('DB_CHARSET', 'utf8mb4'),
        'autocommit': True,
    }
    
    # 验证配置
    if not config['host'] or config['host'].strip() == '':
        raise ValueError(f"数据库 host 配置为空！环境变量 DB_HOST: {os.environ.get('DB_HOST')}")
    if not config['user'] or config['user'].strip() == '':
        raise ValueError(f"数据库 user 配置为空！环境变量 DB_USER: {os.environ.get('DB_USER')}")
    
    return config


def _get_pdd_config() -> Dict[str, Any]:
    """获取 PDD 数据库配置（支持环境变量覆盖）"""
    config = {
        'host': os.environ.get('PDD_DB_HOST', 'localhost'),
        'user': os.environ.get('PDD_DB_USER', 'root'),
        'password': os.environ.get('PDD_DB_PASSWORD', 'Yy123456@'),
        'database': os.environ.get('PDD_DB_DATABASE', 'pdd_report'),
        'port': int(os.environ.get('PDD_DB_PORT', '3306')),
        'charset': os.environ.get('PDD_DB_CHARSET', 'utf8mb4'),
        'autocommit': True,
    }
    
    # 验证配置
    if not config['host'] or config['host'].strip() == '':
        raise ValueError(f"PDD数据库 host 配置为空！环境变量 PDD_DB_HOST: {os.environ.get('PDD_DB_HOST')}")
    if not config['user'] or config['user'].strip() == '':
        raise ValueError(f"PDD数据库 user 配置为空！环境变量 PDD_DB_USER: {os.environ.get('PDD_DB_USER')}")
    
    return config


# 初始化配置（在模块加载时验证）
DB_CONFIG = _get_db_config()
DB_PDD_CONFIG = _get_pdd_config()


