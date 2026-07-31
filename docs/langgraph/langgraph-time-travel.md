# 使用时间旅行

## 概览

LangGraph 通过检查点支持时间旅行功能，包括两个主要能力：

- **重放 (Replay)**：从先前的检查点重试
- **分支 (Fork)**：从先前的检查点通过修改状态进行分支，以探索替代路径

两者都通过从先前的检查点恢复来工作。检查点之前的节点不会重新执行（结果已保存）。检查点之后的节点会重新执行，包括任何 LLM 调用、API 请求和中断。

## 重放

使用先前检查点的配置调用图，以从该点开始重放。重放会重新执行节点——它不仅是从缓存中读取。LLM 调用、API 请求和中断将再次触发，并可能返回不同的结果。从最终检查点（没有 `next` 节点）进行重放是无操作的。

```python
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, NotRequired
from langchain_core.utils.uuid import uuid7

class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]


def generate_topic(state: State):
    return {"topic": "socks in the dryer"}


def write_joke(state: State):
    return {"joke": f"Why do {state['topic']} disappear? They elope!"}


checkpointer = InMemorySaver()
graph = (
    StateGraph(State)
    .add_node("generate_topic", generate_topic)
    .add_node("write_joke", write_joke)
    .add_edge(START, "generate_topic")
    .add_edge("generate_topic", "write_joke")
    .compile(checkpointer=checkpointer)
)

# Step 1: Run the graph
config = {"configurable": {"thread_id": str(uuid7())}}
result = graph.invoke({}, config)

# Step 2: Find a checkpoint to replay from
history = list(graph.get_state_history(config))
# History is in reverse chronological order
for state in history:
    print(f"next={state.next}, checkpoint_id={state.config['configurable']['checkpoint_id']}")

# Step 3: Replay from a specific checkpoint
# Find the checkpoint before write_joke
before_joke = next(s for s in history if s.next == ("write_joke",))
replay_result = graph.invoke(None, before_joke.config)
# write_joke re-executes (runs again), generate_topic does not
```

## 分支 (Fork)

分支从过去的一个检查点创建了一个带有修改状态的新分支。在先前的检查点上调用 `update_state` 来创建分支，然后使用 `None` 调用 `invoke` 以继续执行。

`update_state` 不会回滚线程。它会创建一个从指定点分支出来的新检查点。原始执行历史保持不变。

```python
# Find checkpoint before write_joke
history = list(graph.get_state_history(config))
before_joke = next(s for s in history if s.next == ("write_joke",))

# Fork: update state to change the topic
fork_config = graph.update_state(
    before_joke.config,
    values={"topic": "chickens"},
)

# Resume from the fork — write_joke re-executes with the new topic
fork_result = graph.invoke(None, fork_config)
print(fork_result["joke"])  # A joke about chickens, not socks
```

### 从特定节点

当您调用 `update_state` 时，值会使用指定节点的写入器（包括归约器）进行应用。检查点会记录该节点产生了此次更新，执行将从该节点的后续节点继续。默认情况下，LangGraph 会根据检查点的版本历史推断 `as_node`。当从特定检查点进行分支时，此推断几乎总是正确的。

在以下情况需要明确指定 `as_node`：

- **并行分支**：多个节点在同一步骤中更新了状态，且 LangGraph 无法确定哪个是最后一个（`InvalidUpdateError`）
- **无执行历史**：在新线程上设置状态（在测试中很常见）
- **跳过节点**：将 `as_node` 设置为较晚的节点，让图认为该节点已经运行

```python
# graph: generate_topic -> write_joke

# Treat this update as if generate_topic produced it.
# Execution resumes at write_joke (the successor of generate_topic).
fork_config = graph.update_state(
    before_joke.config,
    values={"topic": "chickens"},
    as_node="generate_topic",
)
```

## 中断

如果您的图在人在回路工作流中使用 `interrupt`，中断在时间旅行期间总是会重新触发。包含中断的节点会重新执行，并且 `interrupt()` 会暂停以等待新的 `Command(resume=...)`。

```python
from langgraph.types import interrupt, Command

class State(TypedDict):
    value: list[str]

def ask_human(state: State):
    answer = interrupt("What is your name?")
    return {"value": [f"Hello, {answer}!"]}

def final_step(state: State):
    return {"value": ["Done"]}

graph = (
    StateGraph(State)
    .add_node("ask_human", ask_human)
    .add_node("final_step", final_step)
    .add_edge(START, "ask_human")
    .add_edge("ask_human", "final_step")
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

# First run: hits interrupt
graph.invoke({"value": []}, config)
# Resume with answer
graph.invoke(Command(resume="Alice"), config)

# Replay from before ask_human
history = list(graph.get_state_history(config))
before_ask = [s for s in history if s.next == ("ask_human",)][-1]

replay_result = graph.invoke(None, before_ask.config)
# Pauses at interrupt — waiting for new Command(resume=...)

# Fork from before ask_human
fork_config = graph.update_state(before_ask.config, {"value": ["forked"]})
fork_result = graph.invoke(None, fork_config)
# Pauses at interrupt — waiting for new Command(resume=...)

# Resume the forked interrupt with a different answer
graph.invoke(Command(resume="Bob"), fork_config)
# Result: {"value": ["forked", "Hello, Bob!", "Done"]}
```

### 多次中断

如果您的图在多个点收集输入（例如，多步骤表单），您可以从中断之间进行分支，以在不重新询问先前问题的情况下更改后续的回答。

```python
def ask_name(state):
    name = interrupt("What is your name?")
    return {"value": [f"name:{name}"]}

def ask_age(state):
    age = interrupt("How old are you?")
    return {"value": [f"age:{age}"]}

# Graph: ask_name -> ask_age -> final
# After completing both interrupts:

# Fork from BETWEEN the two interrupts (after ask_name, before ask_age)
history = list(graph.get_state_history(config))
between = [s for s in history if s.next == ("ask_age",)][-1]

fork_config = graph.update_state(between.config, {"value": ["modified"]})
result = graph.invoke(None, fork_config)
# ask_name result preserved ("name:Alice")
# ask_age pauses at interrupt — waiting for new answer
```

## 子图

使用子图的时间旅行取决于子图是否有自己的检查器。这决定了您可以从中进行时间旅行的检查点的粒度。

### 继承的检查器（默认）

默认情况下，子图继承父级的检查器。父级将整个子图视为一个单一超级步骤——整个子图执行只有一个父级检查点。从子图之前进行时间旅行会从头重新执行它。您无法时间旅行到默认子图中节点之间的点——您只能从父级进行时间旅行。

```python
# Subgraph without its own checkpointer (default)
subgraph = (
    StateGraph(State)
    .add_node("step_a", step_a)       # Has interrupt()
    .add_node("step_b", step_b)       # Has interrupt()
    .add_edge(START, "step_a")
    .add_edge("step_a", "step_b")
    .compile()  # No checkpointer — inherits from parent
)

graph = (
    StateGraph(State)
    .add_node("subgraph_node", subgraph)
    .add_edge(START, "subgraph_node")
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

# Complete both interrupts
graph.invoke({"value": []}, config)            # Hits step_a interrupt
graph.invoke(Command(resume="Alice"), config)  # Hits step_b interrupt
graph.invoke(Command(resume="30"), config)     # Completes

# Time travel from before the subgraph
history = list(graph.get_state_history(config))
before_sub = [s for s in history if s.next == ("subgraph_node",)][-1]

fork_config = graph.update_state(before_sub.config, {"value": ["forked"]})
result = graph.invoke(None, fork_config)
# The entire subgraph re-executes from scratch
# You cannot time travel to a point between step_a and step_b
```

### 子图检查器

在子图上设置 `checkpointer=True` 以赋予其自己的检查点历史。这会在子图内的每个步骤创建检查点，允许您从子图内的特定点进行时间旅行——例如，在两个中断之间。使用带有 `subgraphs=True` 的 `get_state` 来访问子图自己的检查点配置，然后从中进行分支：

```python
# Subgraph with its own checkpointer
subgraph = (
    StateGraph(State)
    .add_node("step_a", step_a)       # Has interrupt()
    .add_node("step_b", step_b)       # Has interrupt()
    .add_edge(START, "step_a")
    .add_edge("step_a", "step_b")
    .compile(checkpointer=True)  # Own checkpoint history
)

graph = (
    StateGraph(State)
    .add_node("subgraph_node", subgraph)
    .add_edge(START, "subgraph_node")
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

# Run until step_a interrupt
graph.invoke({"value": []}, config)

# Resume step_a -> hits step_b interrupt
graph.invoke(Command(resume="Alice"), config)

# Get the subgraph's own checkpoint (between step_a and step_b)
parent_state = graph.get_state(config, subgraphs=True)
sub_config = parent_state.tasks[0].state.config

# Fork from the subgraph checkpoint
fork_config = graph.update_state(sub_config, {"value": ["forked"]})
result = graph.invoke(None, fork_config)
# step_b re-executes, step_a's result is preserved
```

---
来源：https://docs.langchain.org.cn/oss/python/langgraph/use-time-travel
