# 持久化

## 为什么使用持久化

LangGraph 的持久化功能支持以下应用场景：

- **人机回环**：允许人类检查、中断和批准图的步骤
- **记忆**：在交互之间保留对话历史和上下文
- **时光旅行**：回放之前的图执行以调试特定步骤
- **容错性**：从失败的步骤恢复执行
- **待定写入**：保存超级步中已成功完成节点的输出

## 核心概念

### 线程

线程是分配给每个检查点的唯一 ID，包含一系列运行的累积状态。调用图时必须在配置中指定 `thread_id`：

```python
{"configurable": {"thread_id": "1"}}
```

### 检查点

检查点是线程在特定时间点的状态快照，在每个超级步保存。超级步是图的一次"心跳"，其中该步骤计划执行的所有节点都会执行。

#### 代码示例

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": "", "bar":[]}, config)
```

### 检查点命名空间

每个检查点的 `checkpoint_ns` 字段标识其属于哪个图：

- `""`（空字符串）：属于父（根）图
- `"node_name:uuid"`：属于作为节点调用的子图

在节点内访问：

```python
from langchain_core.runnables import RunnableConfig

def my_node(state: State, config: RunnableConfig):
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
```

## 获取与更新状态

### 获取状态

使用 `graph.get_state(config)` 查看最新状态：

```python
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)

# 获取特定检查点的状态
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
graph.get_state(config)
```

返回 `StateSnapshot` 对象示例：

```python
StateSnapshot(
    values={'foo': 'b', 'bar': ['a', 'b']},
    next=(),
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
    metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
    created_at='2024-08-29T19:19:38.821749+00:00',
    parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}}, 
    tasks=()
)
```

### StateSnapshot 字段说明

| 字段 | 类型 | 描述 |
|------|------|------|
| `values` | 字典 | 检查点处的通道状态值 |
| `next` | tuple[str, ...] | 下一个要执行的节点名称，`()` 表示图已完成 |
| `config` | 字典 | 包含 `thread_id`、`checkpoint_ns` 和 `checkpoint_id` |
| `metadata` | 字典 | 执行元数据，含 `source`、`writes` 和 `step` |
| `created_at` | str | 检查点创建时的 ISO 8601 时间戳 |
| `parent_config` | dict \| None | 上一个检查点的配置 |
| `tasks` | tuple[PregelTask, ...] | 此步骤要执行的任务 |

### 获取状态历史

```python
config = {"configurable": {"thread_id": "1"}}
list(graph.get_state_history(config))
```

返回按时间顺序排列的 `StateSnapshot` 列表（最近的在首位）：

```python
[
    StateSnapshot(
        values={'foo': 'b', 'bar': ['a', 'b']},
        next=(),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
        metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
        created_at='2024-08-29T19:19:38.821749+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
        tasks=(),
    ),
    StateSnapshot(
        values={'foo': 'a', 'bar': ['a']},
        next=('node_b',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
        metadata={'source': 'loop', 'writes': {'node_a': {'foo': 'a', 'bar': ['a']}}, 'step': 1},
        created_at='2024-08-29T19:19:38.819946+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
        tasks=(PregelTask(id='6fb7314f-f114-5413-a1f3-d37dfe98ff44', name='node_b', error=None, interrupts=()),),
    ),
    StateSnapshot(
        values={'foo': '', 'bar': []},
        next=('node_a',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
        metadata={'source': 'loop', 'writes': None, 'step': 0},
        created_at='2024-08-29T19:19:38.817813+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
        tasks=(PregelTask(id='f1b14528-5ee5-579c-949b-23ef9bfbed58', name='node_a', error=None, interrupts=()),),
    ),
    StateSnapshot(
        values={'bar': []},
        next=('__start__',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
        metadata={'source': 'input', 'writes': {'foo': ''}, 'step': -1},
        created_at='2024-08-29T19:19:38.816205+00:00',
        parent_config=None,
        tasks=(PregelTask(id='6d27aa2e-d72b-5504-a36f-8620e54a76dd', name='__start__', error=None, interrupts=()),),
    )
]
```

### 查找特定检查点

```python
history = list(graph.get_state_history(config))

# 查找节点执行前的检查点
before_node_b = next(s for s in history if s.next == ("node_b",))

# 按步号查找检查点
step_2 = next(s for s in history if s.metadata["step"] == 2)

# 查找由 update_state 创建的检查点
forks = [s for s in history if s.metadata["source"] == "update"]

# 查找发生中断的检查点
interrupted = next(
    s for s in history
    if s.tasks and any(t.interrupts for t in s.tasks)
)
```

### 重放

重放会从之前的检查点重新执行步骤。使用先前的 `checkpoint_id` 调用图来在该检查点之后重新运行节点。

### 更新状态

```python
graph.update_state(config, {"foo": "updated"}, as_node="node_a")
```

使用 `update_state` 编辑图状态。这会创建带有更新值的新检查点。如果定义了归约器函数，值会通过归约器传递。

## 内存存储 (Memory store)

### 基本用法

```python
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()

user_id = "1"
namespace_for_memory = (user_id, "memories")

import uuid
memory_id = str(uuid.uuid4())
memory = {"food_preference" : "I like pizza"}
store.put(namespace_for_memory, memory_id, memory)

memories = store.search(namespace_for_memory)
memories[-1].dict()
# {'value': {'food_preference': 'I like pizza'},
#  'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
#  'namespace': ['1', 'memories'],
#  'created_at': '2024-10-02T17:22:31.590602+00:00',
#  'updated_at': '2024-10-02T17:22:31.590605+00:00'}
```

### 语义搜索

```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["food_preference", "$"]
    }
)

# 使用自然语言查询查找相关记忆
memories = store.search(
    namespace_for_memory,
    query="What does the user like to eat?",
    limit=3
)
```

存储时指定要嵌入的字段：

```python
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {
        "food_preference": "I love Italian cuisine",
        "context": "Discussing dinner plans"
    },
    index=["food_preference"]
)

# 存储但不嵌入
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {"system_info": "Last updated: 2024-01-01"},
    index=False
)
```

### 在 LangGraph 中使用

编译图时同时使用检查点记录器和存储：

```python
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class Context:
    user_id: str

checkpointer = InMemorySaver()

# ... 定义图 ...

builder = StateGraph(MessagesState, context_schema=Context)
# ... 添加节点和边 ...
graph = builder.compile(checkpointer=checkpointer, store=store)
```

调用图时：

```python
config = {"configurable": {"thread_id": "1"}}

for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi"}]},
    config,
    stream_mode="updates",
    context=Context(user_id="1"),
):
    print(update)
```

在节点中访问存储和用户 ID：

```python
from langgraph.runtime import Runtime
from dataclasses import dataclass
import uuid

@dataclass
class Context:
    user_id: str

async def update_memory(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")
    
    # ... 分析对话并创建新记忆 ...
    
    memory_id = str(uuid.uuid4())
    await runtime.store.aput(namespace, memory_id, {"memory": memory})
```

在任何节点中搜索和使用记忆：

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_id: str

async def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")
    
    memories = await runtime.store.asearch(
        namespace,
        query=state["messages"][-1].content,
        limit=3
    )
    info = "\n".join([d.value["memory"] for d in memories])
    
    # ... 在模型调用中使用记忆 ...
```

在新线程中访问相同记忆（只要 `user_id` 相同）：

```python
config = {"configurable": {"thread_id": "2"}}

for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi, tell me about my memories"}]},
    config,
    stream_mode="updates",
    context=Context(user_id="1"),
):
    print(update)
```

## 优化检查点存储

使用 [`DeltaChannel`](https://reference.langchain.org.cn/python/langgraph/channels/delta/DeltaChannel) 仅存储增量而不是完整的累积值，以减少高频追加通道的检查点大小。

## 检查点记录器库

### 可用实现

- **`langgraph-checkpoint`**：基础接口和内存中实现（`InMemorySaver`），已包含在 LangGraph 中
- **`langgraph-checkpoint-sqlite`**：SQLite 实现（`SqliteSaver` / `AsyncSqliteSaver`），适合实验
- **`langgraph-checkpoint-postgres`**：Postgres 实现（`PostgresSaver` / `AsyncPostgresSaver`），适合生产
- **`langchain-azure-cosmosdb`**：Azure Cosmos DB 实现（`CosmosDBSaverSync` / `CosmosDBSaver`）

### 检查点记录器接口

每个检查点记录器实现以下方法：

- `.put` - 存储带有配置和元数据的检查点
- `.put_writes` - 存储链接到检查点的中间写入
- `.get_tuple` - 使用给定配置获取检查点元组
- `.list` - 列出符合给定条件的检查点

异步版本：`.aput`、`.aput_writes`、`.aget_tuple`、`.alist`

### 序列化器

#### 使用 pickle 进行序列化

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# ... 定义图 ...
graph.compile(
    checkpointer=InMemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
)
```

#### 加密

使用 [`EncryptedSerializer`](https://reference.langchain.org.cn/python/langgraph/checkpoints/#langgraph.checkpoint.serde.encrypted.EncryptedSerializer) 进行加密：

```python
import sqlite3

from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

serde = EncryptedSerializer.from_pycryptodome_aes()  # 读取 LANGGRAPH_AES_KEY
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)
```

```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.postgres import PostgresSaver

serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = PostgresSaver.from_conn_string("postgresql://...", serde=serde)
checkpointer.setup()
```

在 LangSmith 上运行时，只要存在 `LANGGRAPH_AES_KEY` 环境变量，就会自动启用加密。
