# 流式传输

## 概述

LangGraph 实现了一个流式系统来提供实时更新。流式传输对于增强基于LLM构建的应用程序的响应能力至关重要。通过逐步显示输出，甚至在完整响应准备好之前，流式传输显著改善了用户体验 (UX)，特别是在处理LLM的延迟时。

## 入门

### 基本用法

LangGraph 图暴露了 `stream`（同步）和 `astream`（异步）方法，以迭代器形式生成流式输出。传递一个或多个流模式来控制您接收的数据。

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node {node_name} updated: {state}")
    elif chunk["type"] == "custom":
        print(f"Status: {chunk['data']['status']}")
```

**输出**
```
Status: thinking of a joke...
Node generate_joke updated: {'joke': 'Why did the ice cream go to school? To get a sundae education!'}
```

### 完整示例

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer


class State(TypedDict):
    topic: str
    joke: str


def generate_joke(state: State):
    writer = get_stream_writer()
    writer({"status": "thinking of a joke..."})
    return {"joke": f"Why did the {state['topic']} go to school? To get a sundae education!"}

graph = (
    StateGraph(State)
    .add_node(generate_joke)
    .add_edge(START, "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node {node_name} updated: {state}")
    elif chunk["type"] == "custom":
        print(f"Status: {chunk['data']['status']}")
```

**输出**
```
Status: thinking of a joke...
Node generate_joke updated: {'joke': 'Why did the ice cream go to school? To get a sundae education!'}
```

## 流式输出格式 (v2)

需要 LangGraph >= 1.1。本页面上的所有示例都使用 `version="v2"`。

将 `version="v2"` 传递给 `stream()` 或 `astream()` 以获得统一的输出格式。每个块都是一个 `StreamPart` 字典，具有一致的形状——无论流模式、模式数量或子图设置如何。

```python
{
    "type": "values" | "updates" | "messages" | "custom" | "checkpoints" | "tasks" | "debug",
    "ns": (),           # namespace tuple, populated for subgraph events
    "data": ...,        # the actual payload (type varies by stream mode)
}
```

每个流模式都有一个对应的 `TypedDict`。您可以从 `langgraph.types` 导入这些类型。联合类型 `StreamPart` 是 `part["type"]` 上的不相交联合，可以在编辑器和类型检查器中实现完整的类型收窄。

对于 v1（默认），输出格式会根据您的流式传输选项而变化（单一模式返回原始数据，多模式返回 `(mode, data)` 元组，子图返回 `(namespace, data)` 元组）。对于 v2，格式始终相同。

### v2 格式示例

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["values", "updates", "messages", "custom"],
    version="v2",
):
    if chunk["type"] == "values":
        # ValuesStreamPart — full state snapshot after each step
        print(f"State: topic={chunk['data']['topic']}")
    elif chunk["type"] == "updates":
        # UpdatesStreamPart — only the changed keys from each node
        for node_name, state in chunk["data"].items():
            print(f"Node `{node_name}` updated: {state}")
    elif chunk["type"] == "messages":
        # MessagesStreamPart — (message_chunk, metadata) from LLM calls
        msg, metadata = chunk["data"]
        print(msg.content, end="", flush=True)
    elif chunk["type"] == "custom":
        # CustomStreamPart — arbitrary data from get_stream_writer()
        print(f"Progress: {chunk['data']['progress']}%")
```

## 流模式

将以下一个或多个流模式作为列表传递给 `stream()` 或 `astream()` 方法。

| 模式 | 类型 | 描述 |
|------|------|------|
| values | ValuesStreamPart | 每一步后的完整状态。 |
| updates | UpdatesStreamPart | 每一步后的状态更新。同一步骤中的多个更新会单独流式传输。 |
| messages | MessagesStreamPart | 来自LLM调用的 (LLM token, metadata) 2元组。 |
| custom | CustomStreamPart | 通过 `get_stream_writer` 从节点发出的自定义数据。 |
| checkpoints | CheckpointStreamPart | 检查点事件（格式与 `get_state()` 相同）。需要一个检查点器。 |
| tasks | TasksStreamPart | 任务开始/结束事件，包含结果和错误。需要一个检查点器。 |
| debug | DebugStreamPart | 所有可用信息——结合了 `checkpoints` 和 `tasks` 以及额外的元数据。 |

## 图状态

使用流模式 `updates` 和 `values` 在图执行时流式传输图的状态。

- `updates` 在图的每个步骤后流式传输状态的**更新**。
- `values` 在图的每个步骤后流式传输状态的**完整值**。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
  topic: str
  joke: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}


def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
  StateGraph(State)
  .add_node(refine_topic)
  .add_node(generate_joke)
  .add_edge(START, "refine_topic")
  .add_edge("refine_topic", "generate_joke")
  .add_edge("generate_joke", END)
  .compile()
)
```

### 使用 updates 模式

使用此功能可仅流式传输每个步骤后由节点返回的**状态更新**。流式输出包括节点的名称以及更新。

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="updates",
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node `{node_name}` updated: {state}")
```

**输出**
```
Node `refine_topic` updated: {'topic': 'ice cream and cats'}
Node `generate_joke` updated: {'joke': 'This is a joke about ice cream and cats'}
```

### 使用 values 模式

使用此方法在每个步骤之后流式传输图的**完整状态**。

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="values",
    version="v2",
):
    if chunk["type"] == "values":
        print(f"topic: {chunk['data']['topic']}, joke: {chunk['data']['joke']}")
```

**输出**
```
topic: ice cream, joke:
topic: ice cream and cats, joke:
topic: ice cream and cats, joke: This is a joke about ice cream and cats
```

## LLM Tokens

使用 `messages` 流模式从图的任何部分（包括节点、工具、子图或任务）逐令牌流式传输大型语言模型 (LLM) 输出。

来自 `messages` 模式的流式输出是一个元组 `(message_chunk, metadata)`，其中：

- `message_chunk`：来自 LLM 的令牌或消息段。
- `metadata`：一个包含图节点和 LLM 调用详细信息的字典。

> 如果您的LLM未作为LangChain集成提供，您可以使用 `custom` 模式来流式传输其输出。

**Python < 3.11 异步的手动配置要求** 当在 Python < 3.11 中使用异步代码时，您必须显式地将 `RunnableConfig` 传递给 `ainvoke()` 以启用正常的流式传输。

```python
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START


@dataclass
class MyState:
    topic: str
    joke: str = ""


model = init_chat_model(model="gpt-5.4-mini")

def call_model(state: MyState):
    """Call the LLM to generate a joke about a topic"""
    # Note that message events are emitted even when the LLM is run using .invoke rather than .stream
    model_response = model.invoke(
        [
            {"role": "user", "content": f"Generate a joke about {state.topic}"}
        ]
    )
    return {"joke": model_response.content}

graph = (
    StateGraph(MyState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

# The "messages" stream mode streams LLM tokens with metadata
# Use version="v2" for a unified StreamPart format
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        message_chunk, metadata = chunk["data"]
        if message_chunk.content:
            print(message_chunk.content, end="|", flush=True)
```

### 按LLM调用筛选

您可以将 `tags` 与 LLM 调用关联起来，以根据 LLM 调用筛选流式令牌。

```python
from langchain.chat_models import init_chat_model

# model_1 is tagged with "joke"
model_1 = init_chat_model(model="gpt-5.4-mini", tags=['joke'])
# model_2 is tagged with "poem"
model_2 = init_chat_model(model="gpt-5.4-mini", tags=['poem'])

graph = ... # define a graph that uses these LLMs

# The stream_mode is set to "messages" to stream LLM tokens
# The metadata contains information about the LLM invocation, including the tags
async for chunk in graph.astream(
    {"topic": "cats"},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        msg, metadata = chunk["data"]
        # Filter the streamed tokens by the tags field in the metadata to only include
        # the tokens from the LLM invocation with the "joke" tag
        if metadata["tags"] == ["joke"]:
            print(msg.content, end="|", flush=True)
```

### 按LLM调用筛选的扩展示例

```python
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph

# The joke_model is tagged with "joke"
joke_model = init_chat_model(model="gpt-5.4-mini", tags=["joke"])
# The poem_model is tagged with "poem"
poem_model = init_chat_model(model="gpt-5.4-mini", tags=["poem"])


class State(TypedDict):
      topic: str
      joke: str
      poem: str


async def call_model(state, config):
      topic = state["topic"]
      print("Writing joke...")
      # Note: Passing the config through explicitly is required for python < 3.11
      # Since context var support wasn't added before then: https://docs.pythonlang.cn/3/library/asyncio-task.html#creating-tasks
      # The config is passed through explicitly to ensure the context vars are propagated correctly
      # This is required for Python < 3.11 when using async code. Please see the async section for more details
      joke_response = await joke_model.ainvoke(
            [{"role": "user", "content": f"Write a joke about {topic}"}],
            config,
      )
      print("\n\nWriting poem...")
      poem_response = await poem_model.ainvoke(
            [{"role": "user", "content": f"Write a short poem about {topic}"}],
            config,
      )
      return {"joke": joke_response.content, "poem": poem_response.content}


graph = (
      StateGraph(State)
      .add_node(call_model)
      .add_edge(START, "call_model")
      .compile()
)

# The stream_mode is set to "messages" to stream LLM tokens
# The metadata contains information about the LLM invocation, including the tags
async for chunk in graph.astream(
      {"topic": "cats"},
      stream_mode="messages",
      version="v2",
):
    if chunk["type"] == "messages":
        msg, metadata = chunk["data"]
        if metadata["tags"] == ["joke"]:
            print(msg.content, end="|", flush=True)
```

### 从流中省略消息

使用 `nostream` 标签完全将 LLM 输出排除在流之外。带有 `nostream` 标签的调用仍然会运行并生成输出；它们的令牌只是不会在 `messages` 模式下发出。

这在以下情况下很有用：

- 您需要 LLM 输出进行内部处理（例如结构化输出），但不想将其流式传输到客户端
- 您通过不同的通道流式传输相同的内容（例如自定义 UI 消息），并希望避免在 `messages` 流中出现重复输出

```python
from typing import Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import START, StateGraph

stream_model = ChatAnthropic(model_name="claude-haiku-4-5-20251001")
internal_model = ChatAnthropic(model_name="claude-haiku-4-5-20251001").with_config(
    {"tags": ["nostream"]}
)


class State(TypedDict):
    topic: str
    answer: str
    notes: str


def answer(state: State) -> dict[str, Any]:
    r = stream_model.invoke(
        [{"role": "user", "content": f"Reply briefly about {state['topic']}"}]
    )
    return {"answer": r.content}


def internal_notes(state: State) -> dict[str, Any]:
    # Tokens from this model are omitted from stream_mode="messages" because of nostream
    r = internal_model.invoke(
        [{"role": "user", "content": f"Private notes on {state['topic']}"}]
    )
    return {"notes": r.content}


graph = (
    StateGraph(State)
    .add_node("write_answer", answer)
    .add_node("internal_notes", internal_notes)
    .add_edge(START, "write_answer")
    .add_edge("write_answer", "internal_notes")
    .compile()
)

initial_state: State = {"topic": "AI", "answer": "", "notes": ""}
stream = graph.stream(initial_state, stream_mode="messages")
```

### 按节点筛选

若要仅从特定节点流式传输令牌，请使用 `stream_mode="messages"` 并通过流式元数据中的 `langgraph_node` 字段筛选输出。

```python
# The "messages" stream mode streams LLM tokens with metadata
# Use version="v2" for a unified StreamPart format
for chunk in graph.stream(
    inputs,
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        msg, metadata = chunk["data"]
        # Filter the streamed tokens by the langgraph_node field in the metadata
        # to only include the tokens from the specified node
        if msg.content and metadata["langgraph_node"] == "some_node_name":
            ...
```

### 从特定节点流式传输 LLM 令牌的扩展示例

```python
from typing import TypedDict
from langgraph.graph import START, StateGraph
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-5.4-mini")


class State(TypedDict):
      topic: str
      joke: str
      poem: str


def write_joke(state: State):
      topic = state["topic"]
      joke_response = model.invoke(
            [{"role": "user", "content": f"Write a joke about {topic}"}]
      )
      return {"joke": joke_response.content}


def write_poem(state: State):
      topic = state["topic"]
      poem_response = model.invoke(
            [{"role": "user", "content": f"Write a short poem about {topic}"}]
      )
      return {"poem": poem_response.content}


graph = (
      StateGraph(State)
      .add_node(write_joke)
      .add_node(write_poem)
      # write both the joke and the poem concurrently
      .add_edge(START, "write_joke")
      .add_edge(START, "write_poem")
      .compile()
)

# The "messages" stream mode streams LLM tokens with metadata
# Use version="v2" for a unified StreamPart format
for chunk in graph.stream(
    {"topic": "cats"},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        msg, metadata = chunk["data"]
        # Filter the streamed tokens by the langgraph_node field in the metadata
        # to only include the tokens from the write_poem node
        if msg.content and metadata["langgraph_node"] == "write_poem":
            print(msg.content, end="|", flush=True)
```

## 自定义数据

要从 LangGraph 节点或工具内部发送**自定义用户定义数据**，请遵循以下步骤：

1. 使用 `get_stream_writer` 访问流写入器并发出自定义数据。
2. 调用 `.stream()` 或 `.astream()` 时，设置 `stream_mode="custom"` 以在流中获取自定义数据。您可以组合多种模式（例如，`["updates", "custom"]`），但至少一个必须是 `"custom"`。

**Python < 3.11 异步中没有 `get_stream_writer`** 在 Python < 3.11 上运行的异步代码中，`get_stream_writer` 将不起作用。相反，请向您的节点或工具添加一个 `writer` 参数并手动传递它。

### 在节点中使用自定义数据

```python
from typing import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START

class State(TypedDict):
    query: str
    answer: str

def node(state: State):
    # Get the stream writer to send custom data
    writer = get_stream_writer()
    # Emit a custom key-value pair (e.g., progress update)
    writer({"custom_key": "Generating custom data inside node"})
    return {"answer": "some data"}

graph = (
    StateGraph(State)
    .add_node(node)
    .add_edge(START, "node")
    .compile()
)

inputs = {"query": "example"}

# Set stream_mode="custom" to receive the custom data in the stream
for chunk in graph.stream(inputs, stream_mode="custom", version="v2"):
    if chunk["type"] == "custom":
        print(f"Custom event: {chunk['data']['custom_key']}")
```

### 在工具中使用自定义数据

```python
from langchain.tools import tool
from langgraph.config import get_stream_writer

@tool
def query_database(query: str) -> str:
    """Query the database."""
    # Access the stream writer to send custom data
    writer = get_stream_writer()
    # Emit a custom key-value pair (e.g., progress update)
    writer({"data": "Retrieved 0/100 records", "type": "progress"})
    # perform query
    # Emit another custom key-value pair
    writer({"data": "Retrieved 100/100 records", "type": "progress"})
    return "some-answer"


graph = ... # define a graph that uses this tool

# Set stream_mode="custom" to receive the custom data in the stream
for chunk in graph.stream(inputs, stream_mode="custom", version="v2"):
    if chunk["type"] == "custom":
        print(f"{chunk['data']['type']}: {chunk['data']['data']}")
```

## 子图输出

要在流式输出中包含来自子图的输出，您可以在父图的 `.stream()` 方法中设置 `subgraphs=True`。这将同时流式传输父图和任何子图的输出。

输出将以 `StreamPart` 格式流式传输，其中 `ns` 字段是一个元组，包含调用子图的节点的路径，例如 `("parent_node:<task_id>", "child_node:<task_id>")`。

### v2 (LangGraph >= 1.1)

使用 `version="v2"` 时，子图事件使用相同的 `StreamPart` 格式。`ns` 字段标识来源。

```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True,
    stream_mode="updates",
    version="v2",
):
    print(chunk["type"])  # "updates"
    print(chunk["ns"])    # () for root, ("node_name:<task_id>",) for subgraph
    print(chunk["data"])  # {"node_name": {"key": "value"}}
```

### 从子图流式传输的扩展示例

```python
from langgraph.graph import START, StateGraph
from typing import TypedDict

# Define subgraph
class SubgraphState(TypedDict):
    foo: str  # note that this key is shared with the parent graph state
    bar: str

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# Define parent graph
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    # Set subgraphs=True to stream outputs from subgraphs
    subgraphs=True,
    version="v2",
):
    if chunk["type"] == "updates":
        if chunk["ns"]:
            print(f"Subgraph {chunk['ns']}: {chunk['data']}")
        else:
            print(f"Root: {chunk['data']}")
```

**输出**
```
Root: {'node_1': {'foo': 'hi! foo'}}
Subgraph ('node_2:dfddc4ba-c3c5-6887-5012-a243b5b377c2',): {'subgraph_node_1': {'bar': 'bar'}}
Subgraph ('node_2:dfddc4ba-c3c5-6887-5012-a243b5b377c2',): {'subgraph_node_2': {'foo': 'hi! foobar'}}
Root: {'node_2': {'foo': 'hi! foobar'}}
```

**注意**，我们不仅接收节点更新，还接收命名空间，它告诉我们正在从哪个图（或子图）进行流式传输。

## 检查点 (Checkpoints)

使用 `checkpoints` 流模式在图执行时接收检查点事件。每个检查点事件的格式与 `get_state()` 的输出相同。需要一个检查点器。

```python
from langgraph.checkpoint.memory import MemorySaver

graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile(checkpointer=MemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream(
    {"topic": "ice cream"},
    config=config,
    stream_mode="checkpoints",
    version="v2",
):
    if chunk["type"] == "checkpoints":
        print(chunk["data"])
```

## 任务

使用 `tasks` 流模式在图执行时接收任务开始和结束事件。任务事件包括有关哪个节点正在运行、其结果和任何错误的信息。需要一个检查点器。

```python
from langgraph.checkpoint.memory import MemorySaver

graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile(checkpointer=MemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream(
    {"topic": "ice cream"},
    config=config,
    stream_mode="tasks",
    version="v2",
):
    if chunk["type"] == "tasks":
        print(chunk["data"])
```

## 调试

使用 `debug` 流模式在图的整个执行过程中尽可能多地流式传输信息。流式输出包括节点的名称和完整状态。

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="debug",
    version="v2",
):
    if chunk["type"] == "debug":
        print(chunk["data"])
```

`debug` 模式将 `checkpoints` 和 `tasks` 事件与附加元数据相结合。如果您只需要部分调试信息，请直接使用 `checkpoints` 或 `tasks`。

## 同时使用多种模式

您可以将列表作为 `stream_mode` 参数传递，以同时流式传输多种模式。

使用 `version="v2"` 时，每个块都是一个 `StreamPart` 字典。使用 `chunk["type"]` 来区分模式：

```python
for chunk in graph.stream(inputs, stream_mode=["updates", "custom"], version="v2"):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node `{node_name}` updated: {state}")
    elif chunk["type"] == "custom":
        print(f"Custom event: {chunk['data']}")
```

## 高级

### 与任何LLM一起使用

您可以使用 `stream_mode="custom"` 从**任何 LLM API** 流式传输数据——即使该 API 不实现 LangChain 聊天模型接口。这使您可以集成提供自己的流式接口的原始 LLM 客户端或外部服务。

```python
from langgraph.config import get_stream_writer

def call_arbitrary_model(state):
    """Example node that calls an arbitrary model and streams the output"""
    # Get the stream writer to send custom data
    writer = get_stream_writer()
    # Assume you have a streaming client that yields chunks
    # Generate LLM tokens using your custom streaming client
    for chunk in your_custom_streaming_client(state["topic"]):
        # Use the writer to send custom data to the stream
        writer({"custom_llm_chunk": chunk})
    return {"result": "completed"}

graph = (
    StateGraph(State)
    .add_node(call_arbitrary_model)
    # Add other nodes and edges as needed
    .compile()
)
# Set stream_mode="custom" to receive the custom data in the stream
for chunk in graph.stream(
    {"topic": "cats"},
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        # The chunk data will contain the custom data streamed from the llm
        print(chunk["data"])
```

### 流式传输任意聊天模型的扩展示例

```python
import operator
import json

from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START

from openai import AsyncOpenAI

openai_client = AsyncOpenAI()
model_name = "gpt-5.4-mini"


async def stream_tokens(model_name: str, messages: list[dict]):
    response = await openai_client.chat.completions.create(
        messages=messages, model=model_name, stream=True
    )
    role = None
    async for chunk in response:
        delta = chunk.choices[0].delta

        if delta.role is not None:
            role = delta.role

        if delta.content:
            yield {"role": role, "content": delta.content}


# this is our tool
async def get_items(place: str) -> str:
    """Use this tool to list items one might find in a place you're asked about."""
    writer = get_stream_writer()
    response = ""
    async for msg_chunk in stream_tokens(
        model_name,
        [
            {
                "role": "user",
                "content": (
                    "Can you tell me what kind of items "
                    f"i might find in the following place: '{place}'. "
                    "List at least 3 such items separating them by a comma. "
                    "And include a brief description of each item."
                ),
            }
        ],
    ):
        response += msg_chunk["content"]
        writer(msg_chunk)

    return response


class State(TypedDict):
    messages: Annotated[list[dict], operator.add]


# this is the tool-calling graph node
async def call_tool(state: State):
    ai_message = state["messages"][-1]
    tool_call = ai_message["tool_calls"][-1]

    function_name = tool_call["function"]["name"]
    if function_name != "get_items":
        raise ValueError(f"Tool {function_name} not supported")

    function_arguments = tool_call["function"]["arguments"]
    arguments = json.loads(function_arguments)

    function_response = await get_items(**arguments)
    tool_message = {
        "tool_call_id": tool_call["id"],
        "role": "tool",
        "name": function_name,
        "content": function_response,
    }
    return {"messages": [tool_message]}


graph = (
    StateGraph(State)
    .add_node(call_tool)
    .add_edge(START, "call_tool")
    .compile()
)
```

让我们使用包含工具调用的 `AIMessage` 来调用图。

```python
inputs = {
    "messages": [
        {
            "content": None,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "arguments": '{"place":"bedroom"}',
                        "name": "get_items",
                    },
                    "type": "function",
                }
            ],
        }
    ]
}

async for chunk in graph.astream(
    inputs,
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        print(chunk["data"]["content"], end="|", flush=True)
```

### 为特定聊天模型禁用流式传输

如果您的应用程序混合了支持流式传输的模型和不支持流式传输的模型，您可能需要显式禁用对不支持流式传输的模型的流式传输。

#### 使用 init_chat_model

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-6",
    # Set streaming=False to disable streaming for the chat model
    streaming=False
)
```

#### 使用聊天模型接口

```python
from langchain_openai import ChatOpenAI

# Set streaming=False to disable streaming for the chat model
model = ChatOpenAI(model="o1-preview", streaming=False)
```

并非所有聊天模型集成都支持 `streaming` 参数。如果您的模型不支持它，请改用 `disable_streaming=True`。所有聊天模型都通过基类提供此参数。

## 迁移到v2

v2 流式传输格式（本页中使用的）提供统一的输出格式。以下是主要区别和迁移方法的摘要：

| 场景 | v1 (默认) | v2 (`version="v2"`) |
|------|---------|-----------------|
| 单一流模式 | 原始数据 (dict) | 包含 `type`, `ns`, `data` 的 `StreamPart` 字典 |
| 多流模式 | `(mode, data)` 元组 | 相同的 `StreamPart` 字典，基于 `chunk["type"]` 筛选 |
| 子图流式传输 | `(namespace, data)` 元组 | 相同的 `StreamPart` 字典，检查 `chunk["ns"]` |
| 多模式 + 子图 | `(namespace, mode, data)` 三元组 | 相同的 `StreamPart` 字典 |
| `invoke()` 返回类型 | 普通字典 (状态) | 带有 `.value` 和 `.interrupts` 的 `GraphOutput` |
| 中断位置 (流) | 状态字典中的 `__interrupt__` 键 | `values` 流部分上的 `interrupts` 字段 |
| 中断位置 (调用) | 结果字典中的 `__interrupt__` 键 | `GraphOutput` 上的 `.interrupts` 属性 |
| Pydantic/dataclass 输出 | 返回普通字典 | 强制转换为模型/dataclass 实例 |

### v2调用格式

当您将 `version="v2"` 传递给 `invoke()` 或 `ainvoke()` 时，它会返回一个带有 `.value` 和 `.interrupts` 属性的 `GraphOutput` 对象。

```python
from langgraph.types import GraphOutput

result = graph.invoke(inputs, version="v2")

assert isinstance(result, GraphOutput)
result.value       # your output — dict, Pydantic model, or dataclass
result.interrupts  # tuple[Interrupt, ...], empty if none occurred
```

使用除默认 "`values`" 之外的任何流模式时，`invoke(..., stream_mode="updates", version="v2")` 返回 `list[StreamPart]` 而不是 `list[tuple]`。

`GraphOutput` 上的字典式访问（`result["key"]`、`"key" in result`、`result["__interrupt__"]`）仍然有效，以实现向后兼容，但已被**弃用**，并将在未来版本中移除。请迁移到 `result.value` 和 `result.interrupts`。

这将状态与中断元数据分离。

#### v2 示例

```python
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke(inputs, config=config, version="v2")

if result.interrupts:
    print(result.interrupts[0].value)
    graph.invoke(Command(resume=True), config=config, version="v2")
```

### Pydantic和dataclass状态强制转换

当您的图状态是 Pydantic 模型或 dataclass 时，v2 `values` 模式会自动将输出强制转换为正确的类型。

```python
from pydantic import BaseModel
from typing import Annotated
import operator

class MyState(BaseModel):
    value: str
    items: Annotated[list[str], operator.add]

# With version="v2", chunk["data"] is a MyState instance
for chunk in graph.stream(
    {"value": "x", "items": []}, stream_mode="values", version="v2"
):
    print(type(chunk["data"]))  # <class 'MyState'>
```

## Python < 3.11的异步

在 Python < 3.11 版本中，asyncio 任务不支持 `context` 参数。这限制了 LangGraph 自动传播上下文的能力，并以两种关键方式影响 LangGraph 的流式传输机制：

1. 您**必须**显式地将 `RunnableConfig` 传递给异步 LLM 调用（例如，`ainvoke()`），因为回调不会自动传播。
2. 您**不能**在异步节点或工具中使用 `get_stream_writer`——您必须直接传递 `writer` 参数。

### 带有手动配置的异步 LLM 调用

```python
from typing import TypedDict
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model

model = init_chat_model(model="gpt-5.4-mini")

class State(TypedDict):
    topic: str
    joke: str

# Accept config as an argument in the async node function
async def call_model(state, config):
    topic = state["topic"]
    print("Generating joke...")
    # Pass config to model.ainvoke() to ensure proper context propagation
    joke_response = await model.ainvoke(
        [{"role": "user", "content": f"Write a joke about {topic}"}],
        config,
    )
    return {"joke": joke_response.content}

graph = (
    StateGraph(State)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

# Set stream_mode="messages" to stream LLM tokens
async for chunk in graph.astream(
    {"topic": "ice cream"},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        message_chunk, metadata = chunk["data"]
        if message_chunk.content:
            print(message_chunk.content, end="|", flush=True)
```

### 带有流写入器的异步自定义流式传输

```python
from typing import TypedDict
from langgraph.types import StreamWriter

class State(TypedDict):
      topic: str
      joke: str

# Add writer as an argument in the function signature of the async node or tool
# LangGraph will automatically pass the stream writer to the function
async def generate_joke(state: State, writer: StreamWriter):
      writer({"custom_key": "Streaming custom data while generating a joke"})
      return {"joke": f"This is a joke about {state['topic']}"}

graph = (
      StateGraph(State)
      .add_node(generate_joke)
      .add_edge(START, "generate_joke")
      .compile()
)

# Set stream_mode="custom" to receive the custom data in the stream  #
async for chunk in graph.astream(
      {"topic": "ice cream"},
      stream_mode="custom",
      version="v2",
):
      if chunk["type"] == "custom":
          print(chunk["data"])
```

---
来源：https://docs.langchain.org.cn/oss/python/langgraph/streaming
