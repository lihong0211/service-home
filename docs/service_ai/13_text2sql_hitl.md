# Text2SQL + 人工审核（HITL）层

> 文件：`service/ai/text2sql.py`（主）、`service/ai/review.py`（集中审核列表）、`model/ai/agent_review_task.py`（审核记录表）
> 生成日期：2026-08-18

---

## 第一部分：背景演进

**Text-to-SQL 的两个老问题**

自然语言转 SQL（Text2SQL）把"查一下最近一个月哪些保单逾期"这种口语问题直接翻译成可执行的 SQL，省去业务人员学 SQL 的成本。但落到生产环境有两个绕不开的风险：

- **写操作风险**：如果 Agent 生成的不是 `SELECT` 而是 `UPDATE`/`DELETE`，一旦直接执行，出错的代价是真实数据被改坏，且往往不可逆。
- **幻觉 SQL / 幻觉成功**：LLM 有概率"没有真正产出可执行的 SQL，却在最终回答里宣称操作已完成"——这不是拒绝服务的问题，而是更危险的静默失败：用户以为数据改好了，实际上什么都没发生。

**Human-in-the-loop 在 Agent 系统中的必要性**

当一个动作不可逆且有真实副作用时，"要不要执行"这个判断不应该完全交给模型。HITL（Human-in-the-loop）的做法是让 Agent 只负责"生成动作方案"，在真正执行前插入一个人工确认点，人可以批准、编辑后批准、或拒绝。这跟本项目 `service/ai/langchain.py` 里的 HITL 演示图是同一套技术底座（LangGraph 的 `interrupt()`/`Command`/`SqliteSaver`），区别是 Text2SQL 这里的执行节点是真实的数据库写操作，批准即真的落库，不是打印一行字符串的演示。

**核心概念**

1. **generate/review/execute 三段式图**：LangGraph `StateGraph` 把"生成 SQL"“审核决策”“真正执行”拆成三个节点，`review` 节点对写操作调用 `interrupt()` 挂起整个图的执行，等外部调用 `resume` 恢复。
2. **`interrupt()`/`Command(resume=...)`**：LangGraph 的暂停-恢复原语。`interrupt(payload)` 会把图执行暂停在当前节点，把 `payload` 通过 `__interrupt__` 返回给调用方；调用方之后带着同一个 `thread_id` 传入 `Command(resume=value)`，图会从暂停点恢复，`value` 就是 `interrupt()` 调用处的返回值。
3. **枚举值自动纠正**：状态类字段（如"生效/终止/暂停"）经常没有用户口语里说的那个值（"正常"），审核前用代码层确定性检查把 SQL 里的错误枚举值改成数据库里实际存在的值，减少人工审核时因为字面不匹配而误判。

**演进脉络**

| 阶段 | commit | 变化 |
|------|--------|------|
| 早期 | （初版） | Text2SQL 只支持 `SELECT`，写操作直接硬拒绝（`text2sql_run` 里 `_is_read_only_sql` 校验不通过就报错） |
| HITL 接入 | `5c3103c` feat: Text2SQL 联动人工审核（HITL），写操作 SQL 不再硬拒绝 | 新增 `generate → review → execute` 图，写操作走 `interrupt()` 而不是硬拒绝 |
| Bug 修复 | `6cc2e2f` fix: Text2SQL 写操作场景下 Agent 拿不出 SQL 时会编造"已成功执行" | 发现并修复"Agent 声称成功但没有真正产出 SQL"的静默说谎问题 |
| 集中化 | `593b837` feat: Text2SQL 枚举值自动纠正 + 集中人工审核列表页 | 新增 `_fix_enum_values` 枚举纠正；新增 `service/ai/review.py` 把 text2sql 和 hitl 演示图的待审核项统一到一张表、一个列表页 |

**本模块定位**

`service/ai/text2sql.py` 是本项目"让 LLM 安全地操作数据库"的落地方案：只读查询走快速路径直接执行，写操作走 HITL 审核路径，两条路径共用同一套 SQL 生成逻辑，审核环节复用 `service/ai/review.py` 提供的跨业务统一审核列表（同时承载 Text2SQL 写操作和 `langchain.py` 的 HITL 演示图）。

---

## 第二部分：架构剖析

**整体分层**

```
routes/ai.py
  │
  ├── /ai/text2sql            → text2sql_api            （旧接口，只读，写操作硬拒绝）
  ├── /ai/text2sql/run        → text2sql_hitl_api        （新接口，写操作走 HITL）
  ├── /ai/text2sql/resume     → text2sql_resume_api      （批准/编辑/拒绝后恢复执行）
  ├── /ai/review/list         → list_review_tasks_api    （集中审核列表，text2sql + hitl 共用）
  ├── /ai/review/{id}/approve → approve_review_task_api
  └── /ai/review/{id}/reject  → reject_review_task_api

service/ai/text2sql.py
  ├── _generate_sql()          自然语言 → SQL（不执行），create_sql_agent + DashScope
  ├── _fix_enum_values()       审核前对 UPDATE 语句做枚举值确定性纠正
  ├── build_text2sql_graph()   generate → review → execute 的 LangGraph StateGraph
  └── run_text2sql_graph()     图的统一调用入口（首跑 / resume 都走它）

service/ai/review.py
  ├── create_review_task()     interrupt 命中时落一条待审核记录（AgentReviewTask）
  ├── list_review_tasks_api()  GET /ai/review/list
  └── approve/reject_review_task_api()  按 source 分发到 run_text2sql_graph / run_hitl_graph
```

**核心数据流**

*只读查询（新旧接口一致）*：

```
question ──▶ _generate_sql(allow_write=False)
                 │ create_sql_agent 用默认 SQL_PREFIX（其中硬编码了
                 │ "DO NOT make any DML statements"）
                 ▼
              SELECT 语句 ──▶ _execute_select() ──▶ 直接返回结果
```

*写操作：硬拒绝时代（旧）*：

```
question ──▶ _generate_sql() ──▶ 若产出 DML
                                     │
                                     ▼
                          text2sql_run() 里 _is_read_only_sql 校验不通过
                                     │
                                     ▼
                          直接返回 error："仅支持 SELECT 查询，禁止写操作"
                          （SQL 从未落库，用户体验是被拒绝，没有绕过空间）
```

*写操作：HITL 时代（新，`/ai/text2sql/run` + `/ai/text2sql/resume`）*：

```
question ──▶ _t2s_generate（allow_write=True，替换成 _SQL_PREFIX_ALLOW_WRITE 提示词，
              db 用 _ReadOnlySQLDatabase 防止 Agent 在生成阶段自己先执行了一遍）
                 │
                 ▼
           产出 DML 语句（未执行）
                 │
                 ▼
           _t2s_review：_is_read_only_sql 判否 → _fix_enum_values 纠正枚举值
                 │
                 ▼
           interrupt({question, sql, warnings})  ← 图执行在此挂起
                 │                                  同时 create_review_task() 落一条 pending 记录
                 ▼
        【等待人工】前端 /ai/review/list 拉取 → 人工在审核页看到 SQL + 警告
                 │
        approve（原样 / 编辑后）─────────────┐         reject
                 │                          │            │
                 ▼                          ▼            ▼
      /ai/review/{id}/approve      resume=True/编辑后SQL   resume=False
                 │                          │            │
                 └──────────▶ run_text2sql_graph(resume=...) ◀┘
                                     │
                        Command(resume=...) 恢复图执行到 _t2s_review 之后
                                     │
                        approved → goto "execute" → _t2s_execute()
                                     │                    真正 engine.begin() 执行 DML
                        rejected → goto END（error="已拒绝执行该 SQL"）
```

**关键设计原则**

- **生成与执行分离**：`_generate_sql` 只负责产出 SQL 文本，不碰真实数据；真正的写库动作只在 `_t2s_execute`（且必须经过 `approved` 决策）里发生，职责单一，便于审计"谁批准了这条 SQL"。
- **审核态与执行态用同一个 checkpointer 持久化**：`SqliteSaver` 落盘到 `data/checkpoints/text2sql.sqlite`，多 worker 部署下 `interrupt` 后的 `resume` 请求打到另一个进程也能找到暂停状态（若用进程内 `MemorySaver` 则会丢失）。
- **人工看到的 SQL 必须和实际执行的 SQL 一致**：`_t2s_review` 里人工直接批准（没有编辑）时，用的是 `_fix_enum_values` 修正后的 `fixed_sql`，而不是 Agent 原始生成的 `state["sql"]`——避免"人工审的是一个版本，落库的是另一个版本"的错位。

**与行业标准方案对比**

| 维度 | 本地 LangChain SQL Agent + 自建 HITL（本模块） | Vanna.AI | 纯只读 SQL Agent（无写操作能力） | 云厂商 Text2SQL（如 DashScope 数据洞察） |
|------|-----------------------------------------------|----------|-------------------------------|------------------------------------------|
| 写操作支持 | 支持，但强制走人工审核后才执行 | 主要面向只读分析查询，写操作非核心场景 | 不支持，从设计上就杜绝风险 | 通常只读，面向 BI 报表场景 |
| 审核/权限机制 | 自建 `interrupt()`/`Command` 图 + 集中审核列表页，可编辑 SQL 后批准 | 无内置审核流程，依赖上层业务自行包装 | 无需审核（没有写操作可审） | 依赖平台自身权限体系，业务方无法定制审核环节 |
| 部署与数据隐私 | 全本地部署，SQL/数据不出域，Schema 训练数据自管理 | 支持自部署，但核心检索依赖其向量库训练 | 视具体实现而定 | 数据经过云端，需要额外的合规评估 |
| 定制成本 | 高（自己维护提示词、审核状态机、枚举纠正），换取行为完全可控 | 中（现成的 RAG-over-schema 方案，定制空间有限） | 低（逻辑最简单） | 低（开箱即用），但业务定制受限于平台能力 |

**选型建议**：只需要只读查询分析、不接受任何写操作风险的场景，选"纯只读 SQL Agent"最简单可靠；需要支持业务写操作（数据修正、状态变更）但又不能放开自动执行的场景（本项目属于此类），自建 HITL 审核流转是目前综合成本最低的方案；如果没有强隐私/合规约束、只想快速拿到一个开箱即用的自然语言查数工具，Vanna.AI 或云厂商方案能省去大量自建成本。

---

## 第三部分：代码实现深度解析

**核心函数/类清单**

| 名称 | 签名 | 作用 |
|------|------|------|
| `_generate_sql` | `(question: str, model: str = DEFAULT_CHAT_MODEL, allow_write: bool = False) -> dict` | 自然语言 → SQL，不执行。`allow_write=True` 时切换提示词和只读防护 DB |
| `_ReadOnlySQLDatabase` | `class(SQLDatabase)` | 重写 `run()`，生成阶段把 Agent 对 `sql_db_query` 工具的写操作调用拦下来，只允许把 SQL 文本写进最终答案 |
| `_extract_sql_from_answer_text` | `(answer: str) -> str` | 从 Agent 最终回答的 ```` ```sql ``` ```` 代码块里正则兜底抠出 SQL |
| `_fix_enum_values` | `(sql: str) -> tuple[str, list[str]]` | 对 `UPDATE ... SET ... WHERE` 语句里的每个赋值做枚举值核对与纠正，返回改好的 SQL + 警告列表 |
| `_guess_normal_value` | `(distinct_values: set) -> str \| None` | 用正/负向关键词打分，从一列的历史枚举值里猜出"正常"对应的真实值 |
| `_t2s_generate` / `_t2s_review` / `_t2s_execute` | LangGraph 节点函数 | 分别对应生成、审核决策（含 `interrupt()`）、真正执行 |
| `build_text2sql_graph` / `run_text2sql_graph` | `(checkpointer=None)` / `(input_state, thread_id, resume=None) -> dict` | 构建图、统一的首跑/恢复入口 |
| `text2sql_hitl_api` / `text2sql_resume_api` | `(request: Request)` | `/ai/text2sql/run`、`/ai/text2sql/resume` 的路由函数 |

**关键实现细节 1：枚举值自动纠正**

`_fix_enum_values`（第 310-356 行）只处理能用正则稳定解析的 `UPDATE table SET col='val', ... WHERE ...` 结构（`_SQL_UPDATE_SET_PATTERN` + `_SQL_ASSIGNMENT_PATTERN`）。对每个 `col='val'` 赋值：

1. 查这列历史上实际出现过的 distinct 值（`SELECT DISTINCT`，`LIMIT 15`）。
2. 只有当这列像"小基数枚举"（distinct 数量 `0 < n <= 12`）且 `val` 不在其中时才处理，避免对自由文本字段（如姓名、备注）误伤。
3. 如果 `val` 属于口语化的"正常"类词（`_NORMAL_WORDS = {"正常", "normal", "ok", ...}`），调用 `_guess_normal_value` 用 `_POSITIVE_STATUS_KEYWORDS`/`_NEGATIVE_STATUS_KEYWORDS` 关键词对每个候选枚举值打分（命中正向词 +1，命中负向词 -1），分数唯一最高才采用；分数不唯一或全为 0（比如枚举值是纯数字/英文缩写，猜不出语义）就放弃自动纠正，只在 `warnings` 里提示人工。
4. 猜中了就直接把 SQL 文本替换掉——**人工审核时看到的就已经是纠正后的 SQL**，不是"猜测建议"，这样人工只需要判断对不对，不需要自己去核对枚举值。

之所以做成代码层确定性检查而不是只靠提示词约束：提示词里已经反复强调"要匹配列的真实枚举值"，但 `qwen-turbo` 这类小模型执行不稳定，实测会照抄用户的口语措辞。代码层规则不依赖模型能力，行为可预测、可测试。

**关键实现细节 2：HITL 审核流转状态机**

`Text2SqlState`（TypedDict，第 269-278 行）的关键字段：`sql`（当前 SQL 文本）、`decision`（`""`/`"approved"`/`"rejected"`）、`executed`（幂等标记）。状态流转：

```
generate ──▶ review ──┬─(SELECT)───────────────▶ execute（decision=approved）──▶ END
                       │
                       └─(写操作)─▶ interrupt() 挂起
                                      │
                            resume=True         resume="编辑后的SQL"      resume=False/其他假值
                                      │                    │                        │
                              decision=approved     decision=approved        decision=rejected
                              sql=fixed_sql          sql=编辑后的SQL           error="已拒绝执行该 SQL"
                                      │                    │                        │
                                      └────────▶ execute ◀─┘                       END
                                                    │
                                          executed=True，真正 engine.begin() 执行
```

对应 `service/ai/review.py` 里的落地：`AgentReviewTask.status` 三态 `pending → approved/rejected`，`create_review_task` 在命中 `interrupt` 时创建（仅一次，`resume` 请求不会重新创建），`approve_review_task_api`/`reject_review_task_api` 按 `source` 分发到 `run_text2sql_graph`（text2sql）或 `run_hitl_graph`（`langchain.py` 的 HITL 演示图），resume 完成后把结果摘要写回 `task.result`，`status` 置为 `approved`/`rejected`，同一条记录不可重复审批（`task.status != "pending"` 直接 400）。

**关键实现细节 3："Agent 拿不出 SQL 却编造已成功执行" 的根因与修复（`6cc2e2f`）**

*根因*：`create_sql_agent` 默认系统提示词（`SQL_PREFIX`）里硬编码了一行 `"DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)"`。开启 `allow_write=True` 后虽然换成了自定义的 `_SQL_PREFIX_ALLOW_WRITE`（第 104-131 行）明确告知"可以生成 DML，人工会审核"，但 `qwen-turbo` 在实际生成时仍有概率对"执行 DML"抱有保留——具体表现为两种失败模式：

1. Agent 把结论直接写进最终文字回答的 ```` ```sql ``` ```` 代码块里，但没有调用 `sql_db_query` 工具，导致 `intermediate_steps` 里提取不到 SQL（`_extract_sql_from_agent_steps` 返回空）。
2. 更糟的情况：Agent 既没有调用工具、也没有在代码块里给出 SQL，却直接在回答文本里用"已成功更新"/"已插入"这类措辞回复用户，而 `_t2s_review` 在 `sql` 为空时会直接 `Command(goto=END)`，什么都没有执行——**用户看到的是一句"操作成功"的假话**。

*修复方案*（`_generate_sql`，第 155-207 行）分两层防御：

- **兜底提取**（第 197-198 行）：`allow_write=True` 且 `intermediate_steps` 里拿不到 SQL 时，退而用 `_extract_sql_from_answer_text` 正则抠最终回答里最后一个 ```` ```sql ``` ```` 代码块（Agent 常先给分析再给结论，取最后一个更接近真正结论）。这解决了失败模式 1。
- **假成功识别**（第 199-204 行）：如果连兜底提取也拿不到 SQL，且回答文本命中 `_FALSE_SUCCESS_CLAIM_PATTERN`（匹配"已(成功)?(更新/插入/删除/修改/完成)"、"successfully updated" 等措辞），不再把这段话当正常 `answer` 透传给前端，而是转成明确的 `error`："AI 未能给出可执行的 SQL，本次请求未做任何修改，请换个说法重试"。这解决了失败模式 2，把"静默说谎"变成"明确报错"。

**设计决策与取舍**

1. **为什么从硬拒绝写操作改成 HITL 审核（`5c3103c`）**：旧版 `text2sql_run` 对写操作直接返回 error，安全但业务价值有限——很多合理的数据修正需求（改一条保单状态）完全做不了，用户只能回退到手动写 SQL 或找研发帮忙。HITL 把"要不要执行"的决策权交还给人，同时保留了自然语言生成 SQL 的效率，代价是引入了审核延迟和额外的状态管理复杂度（图、checkpointer、审核表），但对于有真实业务价值的写操作场景，这个复杂度是值得的。
2. **`_ReadOnlySQLDatabase` 拦截生成阶段的真实执行**：LangChain 的 `sql_db_query` 工具本身会调用 `db.run()` 真正执行 SQL，Agent 为了验证自己写得对不对经常会主动调用它，哪怕问题是"插入一条数据"（实测复现过：generate 阶段先真跑一次，approve 之后 `_t2s_execute` 又真跑一次，插出两行重复数据）。取舍是牺牲了"Agent 生成阶段能验证 SQL 语法/执行结果"这一便利，换取审核环节的有效性——不这样做，`review` 节点的 `interrupt()` 就形同虚设，数据已经在人工看到之前被改了。
3. **`_t2s_execute` 做幂等防御而不深挖根因（第 388-401 行）**：实测发现 resume 之后这个节点会被调用两次，导致写操作真的执行了两遍。按 LangGraph 官方文档，"中断节点 resume 时从头重新执行"是有记录的行为，但按 `review → execute` 这种 `Command` 动态路由不应该让下游节点也跑两次的官方示例来看，具体是 checkpointer 还是多进程时序导致的没有继续深挖。取舍是：不管根因是什么，"生产系统要能扛住重复执行"本身就是该做的防御（`executed` 幂等标记），跟文档里"副作用回滚"一节 Saga 补偿事务是同一个思路，投入产出比更高。
4. **枚举值猜不准就只给警告，不强行纠正**：`_guess_normal_value` 打分不唯一或全零分时放弃自动纠正。取舍是宁可让人工多看一眼警告，也不冒着把 SQL 改错（比如把"正常"错配到语义相反的枚举值）的风险自动纠正——写操作场景下"改错"比"没帮忙改"代价更高。

---

## 第四部分：应用场景与实战

**核心使用场景**

- 业务人员用自然语言查询 `ai` 库里的数据（"最近一周新增了多少条保单记录"），无需写 SQL，直接走 `/ai/text2sql` 或 `/ai/text2sql/run` 的只读路径。
- 业务人员需要做数据修正（"把某订单的状态改成已完成"），走 `/ai/text2sql/run` → 命中 `interrupt` → 由有权限的人在审核列表页确认/编辑 SQL 后批准。
- 运营/数据同事集中处理所有待审核项（同时包含 Text2SQL 写操作和 `langchain.py` HITL 演示图触发的审核），不需要分别在多个页面里各自查看。

**快速上手**

环境依赖：

```bash
pip install langchain langchain-community langchain-openai langgraph sqlalchemy pymysql
export DASHSCOPE_API_KEY=sk-xxx
```

示例 1：只读查询（旧接口，无需审核）

```python
import requests
resp = requests.post("http://localhost:3000/ai/text2sql", json={
    "question": "查询销售额最高的三个产品",
})
print(resp.json()["data"])  # {"sql": "SELECT ...", "data": [...]}
```

示例 2：写操作走 HITL（新接口，两步：发起 + 审核后 resume）

```python
import requests

# 第一步：发起写操作请求，不传 threadId 会自动生成一个
resp = requests.post("http://localhost:3000/ai/text2sql/run", json={
    "question": "把订单 ID 123 的状态改成已完成",
})
data = resp.json()["data"]
if data["waitingForInput"]:
    thread_id = data["threadId"]
    print("待审核 SQL：", data["interrupt"]["sql"])
    print("警告：", data["interrupt"].get("warnings"))

    # 第二步：人工确认后调用 resume（resume=True 原样批准；传字符串则用编辑后的 SQL 批准；False 拒绝）
    resume_resp = requests.post("http://localhost:3000/ai/text2sql/resume", json={
        "threadId": thread_id,
        "resume": True,
    })
    print(resume_resp.json()["data"])  # {"sql": ..., "data": [{"affected_rows": 1}], ...}
```

示例 3：集中审核列表页（业务方常用来批量处理待审项，而不是记 `threadId` 手工 resume）

```python
import requests

# 拉取所有待审核项（跨 text2sql / hitl 两个来源）
pending = requests.get("http://localhost:3000/ai/review/list", params={"status": "pending"}).json()["data"]["tasks"]
for task in pending:
    print(task["id"], task["source"], task["content"], task.get("warnings"))

# 对某一条按 id 批准（可选传 content 编辑后再批准）
task_id = pending[0]["id"]
requests.post(f"http://localhost:3000/ai/review/{task_id}/approve", json={})
# 或拒绝
# requests.post(f"http://localhost:3000/ai/review/{task_id}/reject")
```

**常见问题排查**

- **写操作请求一直卡在 `waitingForInput: true`**：这是预期行为，不是 bug——写操作必须走 `/ai/review/list` 找到对应任务（按 `threadId` 或 `id`）手动 approve/reject 才会继续。
- **`_generate_sql` 报错"生成 SQL 失败"**：多是 `max_iterations` 步数耗尽（写操作场景 `allow_write=True` 时是 8 步，因为 Agent 常常要先 `SELECT DISTINCT` 核对一次枚举值再写 `UPDATE`），或 DashScope 接口超时（`request_timeout=120`）。
- **审核页看到的 SQL 和自己说的措辞不一致**：多半是命中了 `_fix_enum_values` 的自动纠正，`warnings` 字段会说明原始值和纠正后的值分别是什么，人工确认无误即可批准；如果纠正错了，直接在审核页编辑 SQL 后再批准。
- **同一个 `threadId` 调用 `/ai/text2sql/resume` 两次**：`AgentReviewTask.status != "pending"` 会直接返回 400 "该记录已是 xxx 状态，不能重复审批"，防止重复执行写操作。
- **审核记录不存在（404）**：`task_id` 错误，或者该记录是走 `langchain.py` 的 HITL 演示图产生的（`source=hitl`），确认 `id` 对应的 `source` 字段。

---

## 第五部分：评估与展望

**优势**

- 只读查询和写操作共用同一套 `_generate_sql` 生成逻辑，维护成本低；写操作额外的审核状态机是独立的图节点，不侵入生成逻辑。
- `_fix_enum_values` 把"枚举值不匹配"这类高频错误在人工看到之前就修正好，大幅减少审核环节的来回沟通成本。
- 针对"Agent 编造已成功执行"这种更危险的静默失败做了显式识别和拦截，不是简单地"生成失败就报错"，而是专门识别了"看起来成功、实际什么都没做"的场景。
- 审核列表页跨业务（text2sql、hitl）统一承载，新增一个需要人工审核的 Agent 业务时只需要复用 `create_review_task` + 在 `_resume_by_source` 里加一个分支，不需要各自维护一套审核 UI 和状态。

**已知局限与技术债务**

- **审核带来的延迟**：写操作从发起到真正执行完全依赖人工响应速度，没有超时/自动升级机制；如果审核人长时间不处理，`pending` 记录会一直挂着（`_get_text2sql_checkpointer` 的 SQLite 文件也会持续增长，没有清理策略）。
- **枚举值纠正的边界情况**：`_fix_enum_values` 只处理能被 `_SQL_UPDATE_SET_PATTERN` 正则解析的单表 `UPDATE ... SET ... WHERE` 结构，多表 JOIN 更新、子查询赋值、`INSERT ... SELECT` 等复杂语句不在纠正范围内；`_guess_normal_value` 目前只认识"正常"这一类口语词（`_NORMAL_WORDS`），其它口语化状态描述（比如"催收中"对应"逾期"）无法自动纠正，只能靠人工识别。
- **`_t2s_execute` 重复调用的根因未查清**：目前只是做了幂等防御（`executed` 标记），没有确认是 checkpointer 机制本身还是本项目多进程部署方式导致的，如果后续 LangGraph 版本升级改变了这个行为，幂等标记仍然是必要的兜底，但根因排查仍是技术债务。
- **旧接口 `/ai/text2sql` 与新接口 `/ai/text2sql/run` 并存**：写操作在旧接口上依然是硬拒绝，容易让新老调用方对"写操作到底支不支持"产生困惑，需要在文档/前端层面明确引导用新接口。

**演进建议**

- 短期：给 `pending` 审核项加超时提醒（如企业微信/邮件通知），避免写操作请求被遗忘；`_fix_enum_values` 增加对更多口语状态词的关键词库。
- 中期：审核记录增加"审核人"字段（当前 `AgentReviewTask` 没有记录是谁批准的），便于审计；`data/checkpoints/text2sql.sqlite` 增加定期清理已完成会话的机制。
- 长期：考虑把 `_fix_enum_values` 的正则解析升级为基于 SQL AST 的解析（如 `sqlglot`），覆盖多表更新等当前正则处理不了的场景；评估是否需要给不同业务表/字段配置差异化的审核权限（而不是所有写操作用同一套审核流程）。

**行业前沿**

- **Agent 执行前置校验**：OpenAI/Anthropic 等厂商的 Agent 框架都在推行"计划-确认-执行"的三段式模式（对应本模块的 generate-review-execute），HITL 正在成为有真实副作用的 Agent 系统的标准配置，而不是可选项。
- **结构化输出约束生成阶段的幻觉**：越来越多方案通过 function calling / 强制 JSON schema 而不是纯文本正则解析来获取模型的"结构化决定"，本模块目前依赖正则从回答文本里兜底提取 SQL（`_extract_sql_from_answer_text`），是受限于 `create_sql_agent` 现有工具调用机制的折中方案，未来可评估让 Agent 始终通过工具调用产出 SQL（而不是允许文本兜底）来从根源上减少"文字回答里藏 SQL"的情况。
- **文本转 SQL 的语义层（Semantic Layer）方案**：Vanna.AI、dbt Semantic Layer 等方案把"列名/枚举值/业务术语"的映射关系预先训练/配置好，而不是像本模块这样在生成后才用正则+关键词打分做事后纠正，是更彻底但成本也更高的解法，值得在业务规模变大后评估引入。

---

## 变更记录

- 2026-08-18：从 `08_tools.md` 中独立拆分为专篇，原因是近期 HITL/枚举纠正功能大幅增长，原文档里 Text2SQL 只有寥寥数段（只读白名单机制），已无法覆盖当前 628 行代码里的人工审核流转、枚举值自动纠正、"编造已成功执行" bug 修复等实质内容。
