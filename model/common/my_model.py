# model/common/my_model.py
"""
基础模型类
"""
from datetime import datetime
from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.mysql import INTEGER
from app.app import db


def get_datetime_now():
    return datetime.now()


class MyModel:
    """基础模型类，提供通用的CRUD方法"""
    query = db.session.query_property()
    __tablename__ = "--"
    deleted_at_value = None
    id = Column(INTEGER(11), primary_key=True)
    # 注意：如果数据库表中没有这些字段，需要设置为可选或使用 name 参数映射
    # 如果表中确实没有这些字段，可以注释掉或设置为可选
    create_at = Column(DateTime(), nullable=True, default=get_datetime_now, comment="创建时间")
    update_at = Column(DateTime(), nullable=True, default=get_datetime_now, onupdate=get_datetime_now, comment="修改时间")
    deleted_at = Column(DateTime(), nullable=True, name='deleted_at', comment="删除时间")

    @classmethod
    def insert(cls, json_data, commit=True):
        """插入数据"""
        item = cls.loads(json_data)
        db.session.add(item)
        db.session.flush()
        if commit:
            db.session.commit()
        return item.id

    @classmethod
    def loads(cls, json_data):
        """从字典加载数据到模型"""
        item = cls()
        for key, value in json_data.items():
            # 处理pass字段（Python保留字）
            if key == "pass":
                key = "pass_flag"
            if hasattr(cls, key) and value is not None:
                setattr(item, key, value)
        return item

    @classmethod
    def update(cls, json_data, commit=True):
        """更新数据"""
        update_cols = {}
        for key, value in json_data.items():
            # 处理pass字段（Python保留字）
            if key == "pass":
                key = "pass_flag"
            if hasattr(cls, key) and value is not None and key not in ["id", "create_at", "update_at", "deleted_at"]:
                update_cols[key] = value
        cls.query.where(cls.id == json_data["id"]).update(update_cols)
        if commit:
            db.session.commit()
        return json_data["id"]

    @classmethod
    def batch_update(cls, criterion, update_values, commit=True):
        """批量更新"""
        update_cols = {}
        for key, value in update_values.items():
            if hasattr(cls, key) and value is not None and key not in ["id", "create_at", "update_at", "deleted_at"]:
                update_cols[key] = value
        query = cls.builder_query(criterion)
        query.update(update_cols)
        if commit:
            db.session.commit()

    @classmethod
    def delete(cls, primary_key, commit=True):
        """软删除"""
        update_cols = {cls.deleted_at: datetime.now()}
        cls.query.where(cls.id == primary_key).update(update_cols)
        if commit:
            db.session.commit()

    @classmethod
    def force_delete(cls, criterion, commit=True):
        """硬删除"""
        query = cls.builder_query(criterion)
        query.delete()
        if commit:
            db.session.commit()

    @classmethod
    def get_by_id(cls, primary_key):
        """根据ID获取数据"""
        return cls.query.where(cls.id == primary_key, cls.deleted_at.is_(None)).first()

    @classmethod
    def delete_by(cls, criterion, commit=True):
        """根据条件删除"""
        update_cols = {cls.deleted_at: datetime.now()}
        query = cls.builder_query(criterion)
        query.update(update_cols)
        if commit:
            db.session.commit()

    @classmethod
    def select_by(cls, criterion=None):
        """根据条件查询"""
        if criterion is None:
            criterion = {}
        query = cls.builder_query(criterion)
        return query.all()

    @classmethod
    def builder_query(cls, criterion):
        """构建查询"""
        if criterion is None:
            criterion = {}
        query = cls.query
        order_by_val = None
        for key in criterion:
            val = criterion[key]
            if key == "order_by":
                order_by_val = val
                continue
            elif val is not None:
                if isinstance(val, list):
                    query = query.where(getattr(cls, key).in_(val))
                elif isinstance(val, dict):
                    compare = val["type"]
                    compare_val = val.get("value", None)
                    if compare == "gt":
                        query = query.where(getattr(cls, key) > compare_val)
                    elif compare == "gte":
                        query = query.where(getattr(cls, key) >= compare_val)
                    elif compare == "lt":
                        query = query.where(getattr(cls, key) < compare_val)
                    elif compare == "lte":
                        query = query.where(getattr(cls, key) <= compare_val)
                    elif compare == "like":
                        query = query.where(getattr(cls, key).like(f"%{compare_val}%"))
                    elif compare == "not in":
                        query = query.where(getattr(cls, key).notin_(compare_val))
                    elif compare == "in":
                        query = query.where(getattr(cls, key).in_(compare_val))
                    elif compare == "bt":
                        start = val.get("start", None)
                        end = val.get("end", None)
                        query = query.where(getattr(cls, key) >= start, getattr(cls, key) <= end)
                else:
                    query = query.where(getattr(cls, key) == val)
        if cls.deleted_at_value is None:
            query = query.where(cls.deleted_at.is_(None))
        else:
            query = query.where(cls.deleted_at == cls.deleted_at_value)
        if order_by_val is not None:
            if isinstance(order_by_val, dict):
                col = order_by_val["col"]
                sort = order_by_val["sort"]
                if sort == "desc":
                    query = query.order_by(getattr(cls, col).desc())
                else:
                    query = query.order_by(getattr(cls, col).asc())
            elif isinstance(order_by_val, list):
                for order_item in order_by_val:
                    col = order_item["col"]
                    sort = order_item["sort"]
                    if sort == "desc":
                        query = query.order_by(getattr(cls, col).desc())
                    else:
                        query = query.order_by(getattr(cls, col).asc())
        return query

    @classmethod
    def select_one_by(cls, criterion):
        """根据条件查询单条"""
        query = cls.builder_query(criterion)
        return query.first()

    @classmethod
    def batch_insert(cls, datas, commit=True):
        """批量插入"""
        if len(datas) == 0:
            return
        items = []
        for data in datas:
            item = cls.loads(data)
            items.append(item)
        db.session.add_all(items)
        if commit:
            db.session.commit()

    @classmethod
    def count(cls, criterion=None):
        """统计数量"""
        query = cls.builder_query(criterion)
        # 使用 with_entities 只查询 id，避免查询不存在的字段
        return query.with_entities(cls.id).count()

