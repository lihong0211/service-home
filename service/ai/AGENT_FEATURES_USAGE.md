# Agent 生产化能力 —— 使用说明

这份文档是给"要实际用起来"的人看的：每个新功能在哪个前端页面、怎么点、传什么参数。设计思路和代码定位见
[AGENT_ARCHITECTURE.md](./AGENT_ARCHITECTURE.md)（8 个模块逐一说明）。

前端代码位置：`doctor-dog-web` monorepo 的 `apps/ai`（新增/修改的文件见该仓库对应 commit）。
后端默认地址：开发环境前端走 Vite 代理到 `http://localhost:3001`（见 `apps/ai/vite.config.ts`），
下面 curl 示例以此为准，换成 3000/生产地址同理。

## 1. 上下文感知对话 —— LangChain 页

**路径**：`/skills/langchain`（侧边栏"LangChain"）

- 选中 **Router** 图后正常聊天，不需要做任何额外操作——服务端会自动记住这一次打开页面之后的所有对话，
  刷新页面或点右上角**清空**才会开始一段新会话。
- 每条 AI 回答下面有 👍/👎 两个小图标，点一下就是给这次回答打反馈，会实时变绿/变红。这条反馈会进
  `agent_trace` 表，供 Bad Case 复核页使用。
- **Loop**/**Parallel** 图不支持上下文记忆（这两个是单轮演示图），行为和之前一样。

行为验证：连续问两句「我叫小明」「我刚才说我叫什么名字」，第二句应该能正确回答，且开发者工具网络面板里
看不到 `history` 字段被发送。

## 2. 人工审核（HITL）—— Agent 人工审核 页

**路径**：`/skills/agent-hitl`（侧边栏"人工审核"）

流程：输入一个需要 AI 代办的请求 → AI 生成一条具体建议，暂停等待你审核 → 可以直接改建议内容 →
点「批准并执行」才会真正执行，点「拒绝」则什么都不做。

用来演示：**有副作用的动作不能让模型自己一路跑到底**，这是权限控制里 confirm 档的实际落地。

## 3. Bad Case 复核

**路径**：`/skills/agent-bad-cases`（侧边栏"Bad Case"）

列出所有 `status=error`（系统自己判定失败）或 `feedback=bad`（人工/用户点了 👎）的记录，可以按图名筛选。
这是回流机制的"挖掘"环节——把这张表里的记录人工复核后，该补知识库的补知识库、该改 prompt 的改
prompt、该转成回归测试用例的转测试用例（这三步是人工/离线做的，页面本身不做自动回流）。

## 4. 副作用回滚演示（Saga）

**路径**：`/skills/agent-saga-demo`（侧边栏"副作用回滚"）

两步 Saga：`create_booking`（真实写一行 MySQL `agent_booking` 记录）→ `charge_payment`（可以开关"故意
失败"）。

- 开关打开（默认）：`charge_payment` 抛异常，编排器逆序回滚，`create_booking` 写的那一行会被真实删除——
  页面上两步都会标红，状态显示 `rolled_back`。
- 开关关闭：两步都提交，`agent_booking` 表里会留下这一行，状态显示 `committed`。

用来验证：Agent 执行到一半失败，已经产生的副作用（这里是一行数据库记录）真的能被撤销，不是接口设计
摆设。

## 5. 后端接口速查

给要直接调接口而不经过页面的场景用：

```bash
# 上下文感知对话：只传 threadId，不用传 history
curl -X POST $HOST/ai/langgraph/run \
  -d '{"graph":"router","threadId":"<uuid>","input":{"query":"你好"}}'

# HITL 发起
curl -X POST $HOST/ai/langgraph/run \
  -d '{"graph":"hitl","threadId":"<uuid>","query":"..."}'
# HITL 恢复：resume 传 true=批准 / false=拒绝 / 字符串=编辑后批准
curl -X POST $HOST/ai/langgraph/run \
  -d '{"graph":"hitl","threadId":"<同一个uuid>","resume":true}'

# 给一次回答打反馈（traceId 来自 /ai/langgraph/run 响应里的 data.traceId）
curl -X POST $HOST/ai/langgraph/trace/feedback \
  -d '{"traceId": 1, "rating": "bad", "note": "答非所问"}'

# 查 bad case 池
curl "$HOST/ai/langgraph/trace/bad-cases?graph=router&limit=50"

# Saga 演示
curl -X POST $HOST/ai/langgraph/saga-demo \
  -d '{"item":"预订项目","amount":9900,"failPayment":true}'
```

## 常见问题

**Q: LangChain 页第二轮对话没记住上一轮说的话？**
A: 确认没有点过右上角"清空"（会换新 threadId），且用的是 Router 图（Loop/Parallel 本来就不支持）。

**Q: 反馈按钮点了没反应？**
A: 反馈按钮只在这条回答带 `traceId` 时才出现——如果后端 `agent_trace` 落库失败（比如数据库连不上），
`traceId` 会是 `null`，按钮不会渲染，这是预期行为（宁可不给反馈入口，也不能反馈到一个不存在的记录上）。

**Q: Saga 演示页点执行没反应？**
A: 检查后端服务是否正常运行、`agent_booking` 表是否存在（首次启动服务时会自动建表）。
