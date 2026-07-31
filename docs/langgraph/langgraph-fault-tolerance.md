# 容错性

LangGraph 提供了三种可组合的响应机制来处理节点失败：

## 重试

根据异常类型和退避设置自动重新运行失败的尝试。

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "call_api",
    call_api,
    retry_policy=RetryPolicy(max_attempts=3),
)
```

### 默认行为

`retry_on` 默认使用 `default_retry_on`，会重试除以下异常外的**任何**异常：
- `ValueError`
- `TypeError`
- `ArithmeticError`
- `ImportError`
- `LookupError`
- `NameError`
- `SyntaxError`
- `RuntimeError`
- `ReferenceError`
- `StopIteration`
- `StopAsyncIteration`
- `OSError`

对于 HTTP 库异常，仅在 5xx 状态码时重试。`NodeTimeoutError` 默认可重试。

### 参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|-------|------|
| `max_attempts` | `int` | `3` | 最大尝试次数，包括第一次 |
| `initial_interval` | `float` | `0.5` | 第一次重试前的秒数 |
| `backoff_factor` | `float` | `2.0` | 每次重试后应用于间隔的乘数 |
| `max_interval` | `float` | `128.0` | 重试之间的最大秒数 |
| `jitter` | `bool` | `True` | 向间隔添加随机抖动 |
| `retry_on` | `type[Exception] \| Sequence[type[Exception]] \| Callable[[Exception], bool]` | `default_retry_on` | 要重试的异常或可调用对象 |

### 自定义重试逻辑

```python
from langgraph.types import RetryPolicy, default_retry_on

def custom_retry_on(exc: BaseException) -> bool:
    if isinstance(exc, MyCustomError):
        return False
    return default_retry_on(exc)

builder.add_node(
    "call_api",
    call_api,
    retry_policy=RetryPolicy(max_attempts=3, retry_on=custom_retry_on),
)
```

### 检查重试状态

```python
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

class State(TypedDict):
    result: str

def my_node(state: State, runtime: Runtime) -> State:
    if runtime.execution_info.node_attempt > 1:
        return {"result": call_fallback_api()}
    return {"result": call_primary_api()}

builder = StateGraph(State)
builder.add_node("my_node", my_node, retry_policy=RetryPolicy(max_attempts=3))
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)
```

`execution_info` 暴露以下字段：

| 属性 | 类型 | 描述 |
|------|------|------|
| `node_attempt` | `int` | 当前尝试次数（1-indexed） |
| `node_first_attempt_time` | `float \| None` | 第一次尝试开始时的 Unix 时间戳 |
| `thread_id` | `str \| None` | 当前执行的线程 ID |
| `run_id` | `str \| None` | 当前执行的运行 ID |
| `checkpoint_id` | `str` | 当前执行的检查点 ID |
| `task_id` | `str` | 当前执行的任务 ID |

## 超时

（需要 `langgraph>=1.2`，目前处于 alpha 阶段）

在 `add_node` 上的 `timeout=` 参数限制了单个节点尝试的运行时间。

```python
from datetime import timedelta
from langgraph.types import TimeoutPolicy

# 简单的墙钟上限
builder.add_node("call_model", call_model, timeout=60)
builder.add_node("call_model", call_model, timeout=timedelta(minutes=2))

# 分别设置运行和空闲限制
builder.add_node(
    "call_model",
    call_model,
    timeout=TimeoutPolicy(run_timeout=120, idle_timeout=30),
)
```

节点超时仅适用于**异步**节点。带有 `timeout` 的同步节点会在编译时被拒绝。

### 运行超时

`run_timeout` 是对单次尝试的硬性时钟限制。它从不刷新，无论节点活动如何。

```python
from langgraph.types import TimeoutPolicy

builder.add_node(
    "call_model",
    call_model,
    timeout=TimeoutPolicy(run_timeout=120),
)
```

### 空闲超时

`idle_timeout` 是一个进度重置上限。仅在节点在指定持续时间内停止产生可观察进度时触发。

```python
builder.add_node(
    "call_model",
    call_model,
    timeout=TimeoutPolicy(idle_timeout=30),
)
```

### 进度信号

在默认的 `refresh_on="auto"` 下，空闲计时器在以下任何情况下重置：
- 通过 `CONFIG_KEY_SEND` 进行状态写入
- 流输出（生成的异步流块）
- 子任务调度
- 运行时流写入器调用
- 来自节点或其后代的任何 LangChain 回调事件

### 心跳模式

将 `refresh_on="heartbeat"` 设置为仅将刷新源限制为显式的 `runtime.heartbeat()` 调用。

```python
builder.add_node(
    "call_model",
    call_model,
    timeout=TimeoutPolicy(idle_timeout=30, refresh_on="heartbeat"),
)
```

### 手动心跳

```python
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import TimeoutPolicy
from typing_extensions import TypedDict

class State(TypedDict):
    result: str

async def long_running_node(state: State, runtime: Runtime) -> State:
    for batch in fetch_batches():
        process(batch)
        runtime.heartbeat()
    return {"result": "done"}

builder = StateGraph(State)
builder.add_node(
    "long_running_node",
    long_running_node,
    timeout=TimeoutPolicy(idle_timeout=30, refresh_on="heartbeat"),
)
builder.add_edge(START, "long_running_node")
builder.add_edge("long_running_node", END)
```

### NodeTimeoutError

当超时触发时，LangGraph 会引发 `NodeTimeoutError`。

| 属性 | 类型 | 描述 |
|------|------|------|
| `node` | `str` | 执行超时的节点名称 |
| `elapsed` | `float` | 超时触发前经过的秒数 |
| `kind` | `Literal["idle", "run"]` | 哪个超时被触发 |
| `idle_timeout` | `float \| None` | 配置的空闲超时时间（秒） |
| `run_timeout` | `float \| None` | 配置的运行超时时间（秒） |

`NodeTimeoutError` 默认可重试。结合使用 `timeout=` 与 `retry_policy=`：

```python
from langgraph.types import RetryPolicy, TimeoutPolicy

builder.add_node(
    "call_model",
    call_model,
    timeout=TimeoutPolicy(idle_timeout=30),
    retry_policy=RetryPolicy(max_attempts=3),
)
```

### 使用 Send 实现动态超时

```python
from langgraph.types import Send, TimeoutPolicy

def fan_out(state: OverallState):
    return [
        Send("process_item", {"item": item}, timeout=TimeoutPolicy(idle_timeout=15))
        for item in state["items"]
    ]
```

## 错误处理

（需要 `langgraph>=1.2`，目前处于 alpha 阶段）

错误处理程序在节点失败且所有重试都耗尽后运行。

```python
from langgraph.errors import NodeError
from langgraph.types import Command, RetryPolicy
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict

class State(TypedDict):
    status: str

def charge_payment(state: State) -> State:
    raise RuntimeError("payment gateway timeout")

def payment_error_handler(state: State, error: NodeError) -> Command:
    return Command(
        update={"status": f"compensated: {error.error}"},
        goto="finalize",
    )

def finalize(state: State) -> State:
    return state

graph = (
    StateGraph(State)
    .add_node(
        "charge_payment",
        charge_payment,
        retry_policy=RetryPolicy(max_attempts=3, retry_on=ConnectionError),
        error_handler=payment_error_handler,
    )
    .add_node("finalize", finalize)
    .add_edge(START, "charge_payment")
    .compile()
)
```

### NodeError

```python
from langgraph.errors import NodeError

def my_handler(state: State, error: NodeError) -> Command:
    print(f"Node {error.node} failed with: {error.error}")
    return Command(update={"status": "recovered"}, goto="next_step")
```

| 属性 | 类型 | 描述 |
|------|------|------|
| `node` | `str` | 执行失败的节点名称 |
| `error` | `BaseException` | 失败节点引发的异常 |

### 使用 Command 进行路由

```python
from langgraph.errors import NodeError
from langgraph.types import Command, RetryPolicy
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict

class State(TypedDict):
    status: str

def reserve_inventory(state: State) -> State:
    return {"status": "reserved"}

def charge_payment(state: State) -> State:
    raise RuntimeError("payment timeout")

def payment_error_handler(state: State, error: NodeError) -> Command:
    return Command(
        update={"status": f"compensated_after_{error.node}: {error.error}"},
        goto="finalize",
    )

def finalize(state: State) -> State:
    return state

graph = (
    StateGraph(State)
    .add_node("reserve_inventory", reserve_inventory)
    .add_node(
        "charge_payment",
        charge_payment,
        retry_policy=RetryPolicy(max_attempts=3, retry_on=ConnectionError),
        error_handler=payment_error_handler,
    )
    .add_node("finalize", finalize)
    .add_edge(START, "reserve_inventory")
    .add_edge("reserve_inventory", "charge_payment")
    .compile()
)
```

### 恢复安全的故障

故障来源会被检查点化。如果图在节点失败后但在处理程序完成前被中断，当图从其检查点恢复时，处理程序会看到相同的 `NodeError` 上下文。

### interrupt() 的行为

在节点内部引发的 `interrupt()` **不会**路由到错误处理程序。中断使用 `GraphBubbleUp` 机制来暂停图执行。

### 子图故障

如果一个节点封装了一个子图，并且子图引发了未处理的异常，则该异常会浮出到父节点。父节点的处理程序会使用 `error.error` 中的子图异常来触发。

## 函数式 API

在函数式 API 中，`@task` 和 `@entrypoint` 上也提供了相同的 `timeout=` 和 `retry_policy=` 参数。

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy, TimeoutPolicy

@task(
    timeout=TimeoutPolicy(idle_timeout=30),
    retry_policy=RetryPolicy(max_attempts=3),
)
async def call_api(url: str) -> str:
    response = await fetch(url)
    return response.text

@entrypoint(timeout=60)
async def my_workflow(inputs: dict) -> str:
    result = await call_api("https://api.example.com/data")
    return result
```

## 限制

- **仅限 Python**：JavaScript/TypeScript SDK 中不提供超时和错误处理程序。重试策略在 Python 和 TypeScript 中均可用
- **超时仅适用于异步**：带有 `timeout` 的同步节点会在编译时被拒绝
- **每个节点一个处理程序**：每个节点最多只能有一个 `error_handler`
- **处理程序故障会冒泡**：如果错误处理程序本身引发异常，则该异常会像节点没有处理程序一样传播

---
来源：https://docs.langchain.org.cn/oss/python/langgraph/fault-tolerance
