# EN API - Python版本

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用 Gunicorn 启动服务

```bash
gunicorn -c gunicorn.conf.py wsgi:app
```

## API接口

单词相关

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
