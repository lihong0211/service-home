# 子图

## 概述

子图是在另一个图中用作节点的图。它们适用于构建多智能体系统、复用节点组以及分布式开发场景。

## 设置

```bash
pip install -U langgraph
```

## 定义子图通信

添加子图时需要选择通信模式：

| 模式 | 使用场景 |
|------|---------|
| 状态模式 | 父图和子图有**不同的状态模式**（无共享键），需要状态转换 |
| 添加节点 | 父图和子图**共享状态键**，子图直接读写相同通道 |

## 在节点内调用子图

当父图和子图具有不同的状态模式时，在节点函数内调用子图。

### 基础示例

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class SubgraphState(TypedDict):
    bar: str

# Subgraph
def subgraph_node_1(state: SubgraphState):
    return {"bar": "hi! " + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# Parent graph
class State(TypedDict):
    foo: str

def call_subgraph(state: State):
    # Transform the state to the subgraph state
    subgraph_output = subgraph.invoke({"bar": state["foo"]})
    # Transform response back to the parent state
    return {"foo": subgraph_output["bar"]}

builder = StateGraph(State)
builder.add_node("node_1", call_subgraph)
builder.add_edge(START, "node_1")
graph = builder.compile()
```

### 完整示例：不同的状态模式

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# Define subgraph
class SubgraphState(TypedDict):
    bar: str
    baz: str

def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}

def subgraph_node_2(state: SubgraphState):
    return {"bar": state["bar"] + state["baz"]}

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

def node_2(state: ParentState):
    # Transform the state to the subgraph state
    response = subgraph.invoke({"bar": state["foo"]})
    # Transform response back to the parent state
    return {"foo": response["bar"]}


builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream({"foo": "foo"}, subgraphs=True, version="v2"):
    if chunk["type"] == "updates":
        print(chunk["ns"], chunk["data"])
```

输出：
```
() {'node_1': {'foo': 'hi! foo'}}
('node_2:577b710b-64ae-31fb-9455-6a4d4cc2b0b9',) {'subgraph_node_1': {'baz': 'baz'}}
('node_2:577b710b-64ae-31fb-9455-6a4d4cc2b0b9',) {'subgraph_node_2': {'bar': 'hi! foobaz'}}
() {'node_2': {'foo': 'hi! foobaz'}}
```

### 完整示例：不同的状态模式（两级子图）

```python
# Grandchild graph
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START, END

class GrandChildState(TypedDict):
    my_grandchild_key: str

def grandchild_1(state: GrandChildState) -> GrandChildState:
    return {"my_grandchild_key": state["my_grandchild_key"] + ", how are you"}


grandchild = StateGraph(GrandChildState)
grandchild.add_node("grandchild_1", grandchild_1)

grandchild.add_edge(START, "grandchild_1")
grandchild.add_edge("grandchild_1", END)

grandchild_graph = grandchild.compile()

# Child graph
class ChildState(TypedDict):
    my_child_key: str

def call_grandchild_graph(state: ChildState) -> ChildState:
    grandchild_graph_input = {"my_grandchild_key": state["my_child_key"]}
    grandchild_graph_output = grandchild_graph.invoke(grandchild_graph_input)
    return {"my_child_key": grandchild_graph_output["my_grandchild_key"] + " today?"}

child = StateGraph(ChildState)
child.add_node("child_1", call_grandchild_graph)
child.add_edge(START, "child_1")
child.add_edge("child_1", END)
child_graph = child.compile()

# Parent graph
class ParentState(TypedDict):
    my_key: str

def parent_1(state: ParentState) -> ParentState:
    return {"my_key": "hi " + state["my_key"]}

def parent_2(state: ParentState) -> ParentState:
    return {"my_key": state["my_key"] + " bye!"}

def call_child_graph(state: ParentState) -> ParentState:
    child_graph_input = {"my_child_key": state["my_key"]}
    child_graph_output = child_graph.invoke(child_graph_input)
    return {"my_key": child_graph_output["my_child_key"]}

parent = StateGraph(ParentState)
parent.add_node("parent_1", parent_1)
parent.add_node("child", call_child_graph)
parent.add_node("parent_2", parent_2)

parent.add_edge(START, "parent_1")
parent.add_edge("parent_1", "child")
parent.add_edge("child", "parent_2")
parent.add_edge("parent_2", END)

parent_graph = parent.compile()

for chunk in parent_graph.stream({"my_key": "Bob"}, subgraphs=True, version="v2"):
    if chunk["type"] == "updates":
        print(chunk["ns"], chunk["data"])
```

输出：
```
() {'parent_1': {'my_key': 'hi Bob'}}
('child:2e26e9ce-602f-862c-aa66-1ea5a4655e3b', 'child_1:781bb3b1-3971-84ce-810b-acf819a03f9c') {'grandchild_1': {'my_grandchild_key': 'hi Bob, how are you'}}
('child:2e26e9ce-602f-862c-aa66-1ea5a4655e3b',) {'child_1': {'my_child_key': 'hi Bob, how are you today?'}}
() {'child': {'my_key': 'hi Bob, how are you today?'}}
() {'parent_2': {'my_key': 'hi Bob, how are you today? bye!'}}
```

## 添加子图作为节点

当父图和子图共享状态键时，可以将编译后的子图直接传递给`add_node`。

### 基础示例

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

class State(TypedDict):
    foo: str

# Subgraph
def subgraph_node_1(state: State):
    return {"foo": "hi! " + state["foo"]}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# Parent graph
builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")
graph = builder.compile()
```

### 完整示例：共享状态模式

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# Define subgraph
class SubgraphState(TypedDict):
    foo: str  # shared with parent graph state
    bar: str  # private to SubgraphState

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

for chunk in graph.stream({"foo": "foo"}, version="v2"):
    if chunk["type"] == "updates":
        print(chunk["data"])
```

输出：
```
{'node_1': {'foo': 'hi! foo'}}
{'node_2': {'foo': 'hi! foobar'}}
```

## 子图持久性

使用`.compile()`上的`checkpointer`参数控制子图的持久性：

| 模式 | `检查点器=` | 行为 |
|------|-----------|------|
| 每次调用 | `None`（默认） | 每次调用都从头开始，继承父级检查点器 |
| 每线程 | `True` | 状态在同一线程的不同调用间累积 |
| 无状态 | `False` | 完全没有检查点，像普通函数调用 |

### 有状态

#### 每次调用（默认）

这是大多数应用程序的推荐模式。支持中断、持久执行和并行调用，同时保持每次调用相互隔离。

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

@tool
def fruit_info(fruit_name: str) -> str:
    """Look up fruit info."""
    return f"Info about {fruit_name}"

@tool
def veggie_info(veggie_name: str) -> str:
    """Look up veggie info."""
    return f"Info about {veggie_name}"

# Subagents - no checkpointer setting (inherits parent)
fruit_agent = create_agent(
    model="gpt-5.4-mini",
    tools=[fruit_info],
    prompt="You are a fruit expert. Use the fruit_info tool. Respond in one sentence.",
)

veggie_agent = create_agent(
    model="gpt-5.4-mini",
    tools=[veggie_info],
    prompt="You are a veggie expert. Use the veggie_info tool. Respond in one sentence.",
)

# Wrap subagents as tools for the outer agent
@tool
def ask_fruit_expert(question: str) -> str:
    """Ask the fruit expert. Use for ALL fruit questions."""
    response = fruit_agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
    )
    return response["messages"][-1].content

@tool
def ask_veggie_expert(question: str) -> str:
    """Ask the veggie expert. Use for ALL veggie questions."""
    response = veggie_agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
    )
    return response["messages"][-1].content

# Outer agent with checkpointer
agent = create_agent(
    model="gpt-5.4-mini",
    tools=[ask_fruit_expert, ask_veggie_expert],
    prompt=(
        "You have two experts: ask_fruit_expert and ask_veggie_expert. "
        "ALWAYS delegate questions to the appropriate expert."
    ),
    checkpointer=MemorySaver(),
)
```

##### 中断

每次调用都可以使用`interrupt()`来暂停和恢复。将`interrupt()`添加到工具函数中，以在继续之前需要用户批准：

```python
@tool
def fruit_info(fruit_name: str) -> str:
    """Look up fruit info."""
    interrupt("continue?")
    return f"Info about {fruit_name}"
```

```python
config = {"configurable": {"thread_id": "1"}}

# Invoke - the subagent's tool calls interrupt()
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about apples"}]},
    config=config,
)
# response contains __interrupt__

# Resume - approve the interrupt
response = agent.invoke(Command(resume=True), config=config)
# Subagent message count: 4
```

##### 多轮

每次调用都从一个新的子代理状态开始。子代理不记得以前的调用：

```python
config = {"configurable": {"thread_id": "1"}}

# First call
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about apples"}]},
    config=config,
)
# Subagent message count: 4

# Second call - subagent starts fresh, no memory of apples
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Now tell me about bananas"}]},
    config=config,
)
# Subagent message count: 4 (still fresh!)
```

##### 多次子图调用

对同一子图的多次调用可以无冲突地工作，因为每次调用都有自己的检查点命名空间：

```python
config = {"configurable": {"thread_id": "1"}}

# LLM calls ask_fruit_expert for both apples and bananas
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about apples and bananas"}]},
    config=config,
)
# Subagent message count: 4 (apples - fresh)
# Subagent message count: 4 (bananas - fresh)
```

#### 每线程

当子代理需要记住之前的交互时，请使用按线程持久性。使用`checkpointer=True`进行编译。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

@tool
def fruit_info(fruit_name: str) -> str:
    """Look up fruit info."""
    return f"Info about {fruit_name}"

# Subagent with checkpointer=True for persistent state
fruit_agent = create_agent(
    model="gpt-5.4-mini",
    tools=[fruit_info],
    prompt="You are a fruit expert. Use the fruit_info tool. Respond in one sentence.",
    checkpointer=True,
)

# Wrap subagent as a tool for the outer agent
@tool
def ask_fruit_expert(question: str) -> str:
    """Ask the fruit expert. Use for ALL fruit questions."""
    response = fruit_agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
    )
    return response["messages"][-1].content

# Outer agent with checkpointer
# Use ToolCallLimitMiddleware to prevent parallel calls to per-thread subagents,
# which would cause checkpoint conflicts.
agent = create_agent(
    model="gpt-5.4-mini",
    tools=[ask_fruit_expert],
    prompt="You have a fruit expert. ALWAYS delegate fruit questions to ask_fruit_expert.",
    middleware=[
        ToolCallLimitMiddleware(tool_name="ask_fruit_expert", run_limit=1),
    ],
    checkpointer=MemorySaver(),
)
```

##### 中断

每线程子代理支持`interrupt()`，就像每次调用一样。将`interrupt()`添加到工具函数中以请求用户批准：

```python
@tool
def fruit_info(fruit_name: str) -> str:
    """Look up fruit info."""
    interrupt("continue?")
    return f"Info about {fruit_name}"
```

```python
config = {"configurable": {"thread_id": "1"}}

# Invoke - the subagent's tool calls interrupt()
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about apples"}]},
    config=config,
)
# response contains __interrupt__

# Resume - approve the interrupt
response = agent.invoke(Command(resume=True), config=config)
# Subagent message count: 4
```

##### 多轮

状态在多次调用中累积——子代理记住过去的对话：

```python
config = {"configurable": {"thread_id": "1"}}

# First call
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about apples"}]},
    config=config,
)
# Subagent message count: 4

# Second call - subagent REMEMBERS apples conversation
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Now tell me about bananas"}]},
    config=config,
)
# Subagent message count: 8 (accumulated!)
```

##### 多次子图调用

当您有多个不同的每线程子图时，每个子图都需要自己的存储空间。使用唯一节点名称的`StateGraph`包装器为每个子代理提供稳定的命名空间：

```python
from langgraph.graph import MessagesState, StateGraph

def create_sub_agent(model, *, name, **kwargs):
    """Wrap an agent with a unique node name for namespace isolation."""
    agent = create_agent(model=model, name=name, **kwargs)
    return (
        StateGraph(MessagesState)
        .add_node(name, agent)  # unique name → stable namespace
        .add_edge("__start__", name)
        .compile()
    )

fruit_agent = create_sub_agent(
    "gpt-5.4-mini", name="fruit_agent",
    tools=[fruit_info], prompt="...", checkpointer=True,
)
veggie_agent = create_sub_agent(
    "gpt-5.4-mini", name="veggie_agent",
    tools=[veggie_info], prompt="...", checkpointer=True,
)

config = {"configurable": {"thread_id": "1"}}

# First call - LLM calls both fruit and veggie experts
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about cherries and broccoli"}]},
    config=config,
)
# Fruit subagent message count: 4
# Veggie subagent message count: 4

# Second call - both agents accumulate independently
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Now tell me about oranges and carrots"}]},
    config=config,
)
# Fruit subagent message count: 8 (remembers cherries!)
# Veggie subagent message count: 8 (remembers broccoli!)
```

### 无状态

当您希望子代理像普通函数调用一样运行，没有检查点开销时，使用此选项。使用`checkpointer=False`进行编译：

```python
subgraph_builder = StateGraph(...)
subgraph = subgraph_builder.compile(checkpointer=False)
```

## 检查点器参考

| 功能 | 每次调用（默认） | 每线程 | 无状态 |
|------|-----------------|--------|--------|
| `检查点器=` | `无` | `True` | `False` |
| 中断（HITL） | ✅ | ✅ | ❌ |
| 多轮记忆 | ❌ | ✅ | ❌ |
| 多次调用（不同子图） | ✅ | ⚠️ | ✅ |
| 多次调用（相同子图） | ✅ | ❌ | ✅ |
| 状态检查 | ⚠️ | ✅ | ❌ |

**中断（HITL）**：子图可以使用`interrupt()`暂停执行并等待用户输入，然后从上次中断的地方恢复。

**多轮记忆**：子图在同一线程内的多次调用中保留其状态。每次调用都会从上次中断的地方继续，而不是重新开始。

**多次调用（不同子图）**：可以在单个节点内调用多个不同的子图实例，而不会出现检查点命名空间冲突。

**多次调用（相同子图）**：可以在单个节点内多次调用相同的子图实例。如果是有状态持久性，这些调用会写入相同的检查点命名空间并发生冲突——请改用每次调用持久性。

**状态检查**：子图的状态可以通过`get_state(config, subgraphs=True)`用于调试和监控。

## 查看子图状态

当启用持久性时，可以使用子图选项检查子图状态。

### 每次调用

仅返回当前调用的子图状态。每次调用都从头开始：

```python
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

class State(TypedDict):
    foo: str

# Subgraph
def subgraph_node_1(state: State):
    value = interrupt("Provide value:")
    return {"foo": state["foo"] + value}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()  # inherits parent checkpointer

# Parent graph
builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

graph.invoke({"foo": ""}, config)

# View subgraph state for the current invocation
subgraph_state = graph.get_state(config, subgraphs=True).tasks[0].state

# Resume the subgraph
graph.invoke(Command(resume="bar"), config)
```

### 每线程

返回此线程上所有调用中累积的子图状态：

```python
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# Subgraph with its own persistent state
subgraph_builder = StateGraph(MessagesState)
# ... add nodes and edges
subgraph = subgraph_builder.compile(checkpointer=True)

# Parent graph
builder = StateGraph(MessagesState)
builder.add_node("agent", subgraph)
builder.add_edge(START, "agent")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "what did I say?"}]}, config)

# View accumulated subgraph state (includes messages from both invocations)
subgraph_state = graph.get_state(config, subgraphs=True).tasks[0].state
```

## 流式子图输出

### v2 (LangGraph >= 1.1)

使用`version="v2"`，子图事件使用相同的`StreamPart`格式。`ns`字段标识源图：

```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True,
    stream_mode="updates",
    version="v2",
):
    print(chunk["type"])  # "updates"
    print(chunk["ns"])    # () for root, ("node_2:<task_id>",) for subgraph
    print(chunk["data"])  # {"node_name": {"key": "value"}}
```

### v1 (默认)

```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True,
    stream_mode="updates",
):
    print(chunk)
```

### 从子图流式传输

```python
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START

# Define subgraph
class SubgraphState(TypedDict):
    foo: str
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
    subgraphs=True,
    version="v2",
):
    if chunk["type"] == "updates":
        print(chunk["ns"], chunk["data"])
```

输出：
```
() {'node_1': {'foo': 'hi! foo'}}
('node_2:e58e5673-a661-ebb0-70d4-e298a7fc28b7',) {'subgraph_node_1': {'bar': 'bar'}}
('node_2:e58e5673-a661-ebb0-70d4-e298a7fc28b7',) {'subgraph_node_2': {'foo': 'hi! foobar'}}
() {'node_2': {'foo': 'hi! foobar'}}
```

---
来源：https://docs.langchain.org.cn/oss/python/langgraph/use-subgraphs
