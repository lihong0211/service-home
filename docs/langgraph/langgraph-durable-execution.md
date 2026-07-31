# 持久执行

**持久化执行（Durable execution）** 是一种技术，进程或工作流可以在关键点保存进度，从而允许暂停并在稍后从中断处精确恢复。这在需要 [人机协作（human-in-the-loop）](./langgraph-interrupts.md) 的场景中尤为有用，用户可以在继续之前检查、验证或修改流程；它也适用于可能遇到中断或错误（例如 LLM 调用超时）的长耗时任务。

通过保留已完成的工作，持久化执行使流程能够在无需重新处理先前步骤的情况下恢复——即使是在很长一段时间（例如一周）之后。LangGraph 内置的 [持久化（persistence）](./langgraph-persistence.md) 层为工作流提供了持久化执行能力，确保每个执行步骤的状态都被保存到持久化存储中。该功能保证了如果工作流被中断——无论是由于系统故障还是 [人机协作](./langgraph-interrupts.md) 交互——它都可以从最后记录的状态恢复。

如果您在使用 LangGraph 时配置了检查点（checkpointer），则已默认启用持久化执行。您可以随时暂停和恢复工作流，即使是在中断或故障之后。为了充分利用持久化执行，请确保您的工作流设计是确定性的（deterministic）且幂等的（idempotent），并将任何副作用或非确定性操作封装在任务（tasks）中。您可以在 StateGraph (Graph API) 和函数式 API 中使用这些任务。

## 要求

要在 LangGraph 中利用持久化执行，您需要：

1. 通过指定一个用于保存工作流进度的检查点库（checkpointer）来在工作流中启用[持久化](./langgraph-persistence.md)。
2. 在执行工作流时指定一个线程标识符（thread identifier）。这将跟踪特定工作流实例的执行历史。
3. 将任何非确定性操作（如随机数生成）或具有副作用的操作（如文件写入、API 调用）封装在**任务**中。这可确保工作流恢复时，不会为该次运行重复执行这些操作，而是从持久化层获取结果。

## 确定性与一致性重放

当您恢复工作流运行时，代码并**不会**从停止的**同一行代码**处恢复；相反，它会确定一个合适的起点，从该起点开始继续执行。这意味着工作流将从起点开始重放所有步骤，直到到达停止的位置。

因此，在编写用于持久化执行的工作流时，必须将任何非确定性操作（如随机数生成）和任何具有副作用的操作（如文件写入、API 调用）封装在**任务**或**节点**中。

为确保工作流具有确定性且可一致性地重放，请遵循以下准则：

- **避免重复工作**：如果一个节点包含多个带有副作用的操作（如日志记录、文件写入或网络调用），请将每个操作封装在单独的**任务**中。这确保了当工作流恢复时，这些操作不会重复执行，其结果将从持久化层中检索。
- **封装非确定性操作**：将任何可能产生非确定性结果的代码（如随机数生成）封装在**任务**或**节点**中。这确保了在恢复时，工作流会遵循完全相同的已记录顺序和结果。
- **使用幂等操作**：尽可能确保副作用（如 API 调用、文件写入）是幂等的。这意味着如果操作在工作流故障后重试，其效果与第一次执行时相同。这对于导致数据写入的操作尤为重要。如果一个**任务**开始但未能成功完成，工作流的恢复将重新运行该**任务**，并依赖已记录的结果来保持一致性。请使用幂等键或验证现有结果以避免非预期的重复，从而确保平滑且可预测的工作流执行。

## 持久化模式

LangGraph 支持三种持久化模式，允许您根据应用程序的要求平衡性能和数据一致性。较高的持久化模式会为工作流执行增加更多开销。您可以在调用任何图执行方法时指定持久化模式：

```python
graph.stream(
    {"input": "test"},
    durability="sync"
)
```

持久化模式按持久化程度从低到高排列如下：

- `"exit"`：LangGraph 仅在图执行退出时（无论是成功、出错还是因人机协作中断）才持久化更改。这为长期运行的图提供了最佳性能，但意味着不会保存中间状态，因此无法从执行过程中发生的系统故障（如进程崩溃）中恢复。
- `"async"`：LangGraph 在执行下一步的同时异步持久化更改。这提供了良好的性能和持久性，但如果进程在执行过程中崩溃，存在少量无法写入检查点的风险。
- `"sync"`：LangGraph 在开始下一步之前同步持久化更改。这确保了 LangGraph 在继续执行之前写入每个检查点，以牺牲一定的性能开销为代价，提供了极高的持久性。

## 在节点中使用任务

如果一个节点包含多个操作，您可能会发现将每个操作转换为**任务**比将操作重构为独立节点更容易。

### 原始代码

```python
from typing import NotRequired
from typing_extensions import TypedDict
from langchain_core.utils.uuid import uuid7

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
import requests

# Define a TypedDict to represent the state
class State(TypedDict):
    url: str
    result: NotRequired[str]

def call_api(state: State):
    """Example node that makes an API request."""
    result = requests.get(state['url']).text[:100]  # Side-effect  #
    return {
        "result": result
    }

# Create a StateGraph builder and add a node for the call_api function
builder = StateGraph(State)
builder.add_node("call_api", call_api)

# Connect the start and end nodes to the call_api node
builder.add_edge(START, "call_api")
builder.add_edge("call_api", END)

# Specify a checkpointer
checkpointer = InMemorySaver()

# Compile the graph with the checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Define a config with a thread ID.
thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}

# Invoke the graph
graph.invoke({"url": "https://www.example.com"}, config)
```

### 使用任务

```python
from typing import NotRequired
from typing_extensions import TypedDict
from langchain_core.utils.uuid import uuid7

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import task
from langgraph.graph import StateGraph, START, END
import requests

# Define a TypedDict to represent the state
class State(TypedDict):
    urls: list[str]
    result: NotRequired[list[str]]


@task
def _make_request(url: str):
    """Make a request."""
    return requests.get(url).text[:100]

def call_api(state: State):
    """Example node that makes an API request."""
    requests = [_make_request(url) for url in state['urls']]
    results = [request.result() for request in requests]
    return {
        "results": results
    }

# Create a StateGraph builder and add a node for the call_api function
builder = StateGraph(State)
builder.add_node("call_api", call_api)

# Connect the start and end nodes to the call_api node
builder.add_edge(START, "call_api")
builder.add_edge("call_api", END)

# Specify a checkpointer
checkpointer = InMemorySaver()

# Compile the graph with the checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Define a config with a thread ID.
thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}

# Invoke the graph
graph.invoke({"urls": ["https://www.example.com"]}, config)
```

## 恢复工作流

一旦在工作流中启用了持久化执行，您可以在以下场景中恢复执行：

- **暂停和恢复工作流：** 使用 `interrupt` 函数在特定点暂停工作流，并使用 `Command` 原语使用更新后的状态恢复它。详见[中断（Interrupts）](./langgraph-interrupts.md)。
- **从故障中恢复：** 在异常（例如 LLM 提供商中断）发生后，自动从上一个成功的检查点恢复工作流。这通过提供 `None` 作为输入值并使用相同的线程标识符来执行工作流。

## 恢复工作流的起点

- 如果您使用的是 StateGraph (Graph API)，起点是执行停止所在的**节点（node）**的开头。
- 如果您在节点内进行子图调用，起点将是调用该子图并被挂起的**父节点（parent node）**。在子图内部，起点将是执行停止所在的特定**节点**。
- 如果您使用的是函数式 API，起点是执行停止所在的**入口点（entrypoint）**的开头。

## 优雅停机

需要 `langgraph>=1.2`，目前处于 alpha 阶段。

优雅停机（Graceful shutdown）允许您在当前超步骤（superstep）完成后协作地停止正在进行的图运行，并保存可恢复的检查点。这对于处理 SIGTERM 信号或任何需要回收资源而又不丢失工作的外部监控器非常有用。

创建一个 `RunControl` 并将其作为 `control=` 参数传递给 `invoke` 或 `stream`。从任何线程调用 `request_drain()` 以发出应停止运行的信号：

```python
from langgraph.runtime import RunControl
from langgraph.errors import GraphDrained

control = RunControl()

# In a signal handler or supervisor:
# control.request_drain("sigterm")

try:
    result = graph.invoke(inputs, config, control=control)
except GraphDrained as e:
    # The graph stopped early and saved a checkpoint.
    # Resume later with the same config.
    print(f"Drained: {e.reason}")
```

### 语义

清理（Drain）是协作式的，发生在超步骤之间，绝不会中断正在运行的工作。

| 场景 | 行为 |
|------|------|
| 节点执行中 | 运行直至完成。清理在下一个超步骤生效。 |
| 带有重试策略的节点正在重试中 | 重试循环直至耗尽或成功。清理随后生效。 |
| 图在清理的同时自然完成 | 正常返回。检查 `control.drain_requested` 以区分正常运行。 |
| 仍有超步骤剩余 | 引发 `GraphDrained(reason)`。检查点已保存且可恢复。 |
| 子图请求清理 | `GraphDrained` 会向上传递给父节点，并在其下一个超步骤边界处停止父节点。 |

### 清理（Drain）后恢复

使用相同的 `thread_id` 调用 `invoke(None, config)` 来恢复已清理的运行。

```python
result = graph.invoke(None, config)
```

### 在节点内读取清理状态

通过 `runtime` 参数访问清理状态，以在到达超步骤边界之前调整节点行为。

```python
from langgraph.runtime import Runtime

async def my_node(state: State, runtime: Runtime) -> State:
    if runtime.drain_requested:
        # Skip expensive work and return a minimal result
        return {"status": "skipped", "reason": runtime.drain_reason}
    return {"status": await do_work()}
```

### SIGTERM 钩子模式

处理进程关闭的推荐模式：

```python
import signal
from langgraph.runtime import RunControl
from langgraph.errors import GraphDrained

control = RunControl()
signal.signal(signal.SIGTERM, lambda *_: control.request_drain("sigterm"))

try:
    result = graph.invoke(inputs, config, control=control)
except GraphDrained as e:
    log.info("graph drained: %s", e.reason)
    # Resume on next startup with the same config
```

`request_drain()` 不会取消正在运行的 asyncio 任务或杀死线程。为了获得硬性上限，请将清理与优雅的超时和任务取消结合使用。

---
来源：https://docs.langchain.org.cn/oss/python/langgraph/durable-execution
