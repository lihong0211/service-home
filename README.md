# EN API - Python版本

这是原Node.js/Express项目的Python重构版本，使用Flask框架和SQLAlchemy ORM实现。

## 技术栈

- **Flask**: Web框架
- **SQLAlchemy**: ORM框架，不再手写SQL
- **Flask-SQLAlchemy**: Flask的SQLAlchemy扩展
- **PyMySQL**: MySQL数据库驱动

## 项目结构

```
python/
├── app.py                 # 主应用文件
├── config/                # 配置文件
│   └── db.py             # 数据库配置
├── routes/                # 路由模块
│   └── __init__.py       # 路由定义
├── service/               # 服务层
│   ├── words/            # 单词服务
│   ├── root/             # 词根服务
│   ├── affix/            # 词缀服务
│   ├── dialogue/         # 对话服务
│   ├── livingSpeech/      # 生活用语服务
│   └── peach/            # Peach相关服务
│       ├── pddReport/    # 拼多多报告
│       ├── version/      # 版本管理
│       ├── aliReport/    # 阿里报告
│       ├── check/        # 检查服务
│       ├── config/       # 配置服务
│       └── pluginStatistic/  # 插件统计
├── utils/                 # 工具函数
│   ├── __init__.py       # 工具函数
│   └── db_pool.py        # 数据库连接池
└── requirements.txt       # 依赖包列表
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置数据库

编辑 `config/db.py` 文件，修改数据库连接配置：

```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',
    'database': 'english',
    'port': 3306,
    'charset': 'utf8mb4',
    'autocommit': True,
}
```

## 运行项目

```bash
python app.py
```

或者使用环境变量指定端口：

```bash
PORT=3000 python app.py
```

## API接口

所有API接口与原Node.js版本保持一致，包括：

### 单词相关
- POST `/words/add` - 添加单词
- POST `/words/delete` - 删除单词
- POST `/words/update` - 更新单词
- POST `/words/list` - 查询单词列表

### 词根相关
- POST `/root/add` - 添加词根
- POST `/root/delete` - 删除词根
- POST `/root/update` - 更新词根
- POST `/root/list` - 查询词根列表

### 词缀相关
- POST `/affix/add` - 添加词缀
- POST `/affix/delete` - 删除词缀
- POST `/affix/update` - 更新词缀
- GET `/affix/list` - 查询词缀列表

### 对话相关
- POST `/dialogue/add` - 添加对话
- POST `/dialogue/delete` - 删除对话
- POST `/dialogue/update` - 更新对话
- GET `/dialogue/list` - 查询对话列表

### 生活用语相关
- POST `/living-speech/add` - 添加生活用语
- POST `/living-speech/delete` - 删除生活用语
- POST `/living-speech/update` - 更新生活用语
- POST `/living-speech/list` - 查询生活用语列表

### 拼多多报告相关
- POST `/pddReport/chat/add` - 添加聊天记录
- POST `/pddReport/chat/list` - 查询聊天记录列表
- POST `/pddReport/rp/add` - 添加处方记录
- POST `/pddReport/rp/list` - 查询处方记录列表
- POST `/pddReport/manual/add` - 添加手动记录
- POST `/pddReport/manual/list` - 查询手动记录列表

### 版本相关
- POST `/jdReport/version/add` - 添加版本
- POST `/jdReport/version/list` - 查询版本列表
- POST `/peach/version/add` - 添加版本

### 阿里报告相关
- POST `/aliReport/rp/add` - 添加阿里报告
- POST `/aliReport/rp/get` - 获取阿里报告
- POST `/aliReport/rp/update` - 更新阿里报告

### 检查相关
- POST `/peach/check/add` - 添加检查记录

### 配置相关
- GET `/peach/config/list` - 查询配置列表

### 插件统计相关
- POST `/peach/plugin-statistics/add` - 添加插件统计
- POST `/peach/plugin-statistics/list` - 查询插件统计列表
- POST `/peach/plugin-statistics/detail` - 查询插件统计详情

## ORM模式说明

本项目使用SQLAlchemy ORM模式，不再手写SQL语句：

1. **模型定义**: 所有数据表在 `model/` 目录下定义为ORM模型类
2. **基础模型**: `model/common/my_model.py` 提供了通用的CRUD方法：
   - `insert()`: 插入数据
   - `update()`: 更新数据
   - `delete()`: 软删除（设置deleted_at）
   - `force_delete()`: 硬删除
   - `get_by_id()`: 根据ID查询
   - `select_by()`: 根据条件查询
   - `count()`: 统计数量
   - `builder_query()`: 构建复杂查询

3. **服务层**: 所有服务层代码使用ORM方法操作数据库，不再使用原始SQL

## 注意事项

1. 确保MySQL数据库已创建并配置正确
2. 数据库表结构需要与原Node.js版本保持一致
3. 所有API接口的请求和响应格式与原版本保持一致
4. 首次运行时会自动创建数据库表（如果表不存在）
5. 使用软删除机制，删除操作会设置 `deleted_at` 字段而不是真正删除数据

