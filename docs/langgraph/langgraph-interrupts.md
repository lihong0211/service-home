# 中断

## 概述

中断允许在图执行的特定点暂停，并等待外部输入后继续。这实现了"需要外部输入才能继续的人机交互模式"。LangGraph使用持久化层保存图状态，直到恢复执行。

## 使用 `interrupt` 暂停

`interrupt` 函数会暂停图执行并向调用者返回一个值。使用中断需要：

1. 检查点记录器来持久化图状态
2. 配置中的线程 ID
3. 在暂停位置调用 `interrupt()`

```python
from langgraph.types import interrupt

def approval_node(state: State):
    # Pause and ask for approval
    approved = interrupt("Do you approve this action?")
    
    # When you resume, Command(resume=...) returns that value here
    return {"approved": approved}
```

## 恢复中断

通过再次调用图并包含 `Command` 来恢复：

### v2 (LangGraph >= 1.1)

```python
from langgraph.types import Command

# Initial run - hits the interrupt and pauses
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"input": "data"}, config=config, version="v2")

# result is a GraphOutput with .value and .interrupts
print(result.interrupts)
# > (Interrupt(value='Do you approve this action?'),)

# Resume with the human's response
graph.invoke(Command(resume=True), config=config, version="v2")
```

### v1 (默认)

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"input": "data"}, config=config)

# __interrupt__ contains the payload
print(result["__interrupt__"])
# > [Interrupt(value='Do you approve this action?')]

# Resume with the human's response
graph.invoke(Command(resume=True), config=config)
```

## 关键点

- 恢复时必须使用相同的线程 ID
- 传递给 `Command(resume=...)` 的值成为 `interrupt()` 的返回值
- 恢复时，节点从开头重新开始
- 任何可 JSON 序列化的值都可作为恢复值传递

## 常见模式

### 结合人机交互 (HITL) 中断进行流式传输

```python
async for chunk in graph.astream(
    initial_input,
    stream_mode=["messages", "updates"],
    subgraphs=True,
    config=config,
    version="v2",
):
    if chunk["type"] == "messages":
        # Handle streaming message content
        msg, _ = chunk["data"]
        if isinstance(msg, AIMessageChunk) and msg.content:
            display_streaming_content(msg.content)

    elif chunk["type"] == "updates":
        # Check for interrupts in the updates data
        if "__interrupt__" in chunk["data"]:
            interrupt_info = chunk["data"]["__interrupt__"][0].value
            user_response = get_user_input(interrupt_info)
            initial_input = Command(resume=user_response)
            break
        else:
            current_node = list(chunk["data"].keys())[0]
```

### 处理多个中断

```python
from typing import Annotated, TypedDict
import operator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    vals: Annotated[list[str], operator.add]


def node_a(state):
    answer = interrupt("question_a")
    return {"vals": [f"a:{answer}"]}


def node_b(state):
    answer = interrupt("question_b")
    return {"vals": [f"b:{answer}"]}


graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge(START, "b")
    .add_edge("a", END)
    .add_edge("b", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

# Step 1: invoke - both parallel nodes hit interrupt() and pause
interrupted_result = graph.invoke({"vals": []}, config)
print(interrupted_result)
"""
{
    'vals': [],
    '__interrupt__': [
        Interrupt(value='question_a', id='bd4f3183600f2c41dddafbf8f0f7be7b'),
        Interrupt(value='question_b', id='29963e3d3585f0cef025dd0f14323f55')
    ]
}
"""

# Step 2: resume all pending interrupts at once
resume_map = {
    i.id: f"answer for {i.value}"
    for i in interrupted_result["__interrupt__"]
}
result = graph.invoke(Command(resume=resume_map), config)

print("Final state:", result)
#> Final state: {'vals': ['a:answer for question_a', 'b:answer for question_b']}
```

### 批准或拒绝

```python
from typing import Literal
from langgraph.types import interrupt, Command

def approval_node(state: State) -> Command[Literal["proceed", "cancel"]]:
    # Pause execution; payload shows up under result["__interrupt__"]
    is_approved = interrupt({
        "question": "Do you want to proceed with this action?",
        "details": state["action_details"]
    })
    
    # Route based on the response
    if is_approved:
        return Command(goto="proceed")
    else:
        return Command(goto="cancel")
```

恢复调用：

```python
# To approve
graph.invoke(Command(resume=True), config=config)

# To reject
graph.invoke(Command(resume=False), config=config)
```

完整示例：

```python
from typing import Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    action_details: str
    status: Optional[Literal["pending", "approved", "rejected"]]


def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
    # Expose details so the caller can render them in a UI
    decision = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"],
    })
    
    # Route to the appropriate node after resume
    return Command(goto="proceed" if decision else "cancel")


def proceed_node(state: ApprovalState):
    return {"status": "approved"}


def cancel_node(state: ApprovalState):
    return {"status": "rejected"}


builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)
builder.add_edge(START, "approval")
builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

# Use a more durable checkpointer in production
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "approval-123"}}
initial = graph.invoke(
    {"action_details": "Transfer $500", "status": "pending"},
    config=config,
)
print(initial["__interrupt__"])  # -> [Interrupt(value={'question': ..., 'details': ...})]

# Resume with the decision; True routes to proceed, False to cancel
resumed = graph.invoke(Command(resume=True), config=config)
print(resumed["status"])  # -> "approved"
```

### 审阅并编辑状态

```python
from langgraph.types import interrupt

def review_node(state: State):
    # Pause and show the current content for review
    edited_content = interrupt({
        "instruction": "Review and edit this content",
        "content": state["generated_text"]
    })
    
    # Update the state with the edited version
    return {"generated_text": edited_content}
```

恢复时，提供编辑后的内容：

```python
graph.invoke(
    Command(resume="The edited and improved text"),
    config=config
)
```

完整示例：

```python
import sqlite3
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class ReviewState(TypedDict):
    generated_text: str


def review_node(state: ReviewState):
    # Ask a reviewer to edit the generated content
    updated = interrupt({
        "instruction": "Review and edit this content",
        "content": state["generated_text"],
    })
    return {"generated_text": updated}


builder = StateGraph(ReviewState)
builder.add_node("review", review_node)
builder.add_edge(START, "review")
builder.add_edge("review", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "review-42"}}
initial = graph.invoke({"generated_text": "Initial draft"}, config=config)
print(initial["__interrupt__"])  # -> [Interrupt(value={'instruction': ..., 'content': ...})]

# Resume with the edited text from the reviewer
final_state = graph.invoke(
    Command(resume="Improved draft after review"),
    config=config,
)
print(final_state["generated_text"])  # -> "Improved draft after review"
```

### 工具中的中断

```python
from langchain.tools import tool
from langgraph.types import interrupt

@tool
def send_email(to: str, subject: str, body: str):
    """Send an email to a recipient."""

    # Pause before sending; payload surfaces in result["__interrupt__"]
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Approve sending this email?"
    })

    if response.get("action") == "approve":
        # Resume value can override inputs before executing
        final_to = response.get("to", to)
        final_subject = response.get("subject", subject)
        final_body = response.get("body", body)
        return f"Email sent to {final_to} with subject '{final_subject}'"
    return "Email cancelled by user"
```

完整示例：

```python
import sqlite3
from typing import TypedDict

from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class AgentState(TypedDict):
    messages: list[dict]


@tool
def send_email(to: str, subject: str, body: str):
    """Send an email to a recipient."""

    # Pause before sending; payload surfaces in result["__interrupt__"]
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Approve sending this email?",
    })

    if response.get("action") == "approve":
        final_to = response.get("to", to)
        final_subject = response.get("subject", subject)
        final_body = response.get("body", body)

        # Actually send the email (your implementation here)
        print(f"[send_email] to={final_to} subject={final_subject} body={final_body}")
        return f"Email sent to {final_to}"

    return "Email cancelled by user"


model = ChatAnthropic(model="claude-sonnet-4-6").bind_tools([send_email])


def agent_node(state: AgentState):
    # LLM may decide to call the tool; interrupt pauses before sending
    result = model.invoke(state["messages"])
    return {"messages": state["messages"] + [result]}


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)

checkpointer = SqliteSaver(sqlite3.connect("tool-approval.db"))
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "email-workflow"}}
initial = graph.invoke(
    {
        "messages": [
            {"role": "user", "content": "Send an email to alice@example.com about the meeting"}
        ]
    },
    config=config,
)
print(initial["__interrupt__"])  # -> [Interrupt(value={'action': 'send_email', ...})]

# Resume with approval and optionally edited arguments
resumed = graph.invoke(
    Command(resume={"action": "approve", "subject": "Updated subject"}),
    config=config,
)
print(resumed["messages"][-1])  # -> Tool result returned by send_email
```

### 验证人工输入

```python
from langgraph.types import interrupt

def get_age_node(state: State):
    prompt = "What is your age?"

    while True:
        answer = interrupt(prompt)  # payload surfaces in result["__interrupt__"]

        # Validate the input
        if isinstance(answer, int) and answer > 0:
            # Valid input - continue
            break
        else:
            # Invalid input - ask again with a more specific prompt
            prompt = f"'{answer}' is not a valid age. Please enter a positive number."

    return {"age": answer}
```

完整示例：

```python
import sqlite3
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt


class FormState(TypedDict):
    age: int | None


def get_age_node(state: FormState):
    prompt = "What is your age?"

    while True:
        answer = interrupt(prompt)  # payload surfaces in result["__interrupt__"]

        if isinstance(answer, int) and answer > 0:
            return {"age": answer}

        prompt = f"'{answer}' is not a valid age. Please enter a positive number."


builder = StateGraph(FormState)
builder.add_node("collect_age", get_age_node)
builder.add_edge(START, "collect_age")
builder.add_edge("collect_age", END)

checkpointer = SqliteSaver(sqlite3.connect("forms.db"))
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "form-1"}}
first = graph.invoke({"age": None}, config=config)
print(first["__interrupt__"])  # -> [Interrupt(value='What is your age?', ...)]

# Provide invalid data; the node re-prompts
retry = graph.invoke(Command(resume="thirty"), config=config)
print(retry["__interrupt__"])  # -> [Interrupt(value="'thirty' is not a valid age...", ...)]

# Provide valid data; loop exits and state updates
final = graph.invoke(Command(resume=30), config=config)
print(final["age"])  # -> 30
```

## 中断规则

### 不要将 `interrupt` 调用包裹在 try/except 中

`interrupt` 通过抛出特殊异常来暂停执行。如果包裹在 try/except 中，将捕获此异常，中断不会传回。

✅ 好的做法 - 将 `interrupt` 调用与易错代码分开：

```python
def node_a(state: State):
    # ✅ Good: interrupting first, then handling
    # error conditions separately
    interrupt("What's your name?")
    try:
        fetch_data()  # This can fail
    except Exception as e:
        print(e)
    return state
```

❌ 不好的做法 - 用空的 try/except 包裹 `interrupt`：

```python
def node_a(state: State):
    # ❌ Bad: wrapping interrupt in bare try/except
    # will catch the interrupt exception
    try:
        interrupt("What's your name?")
    except Exception as e:
        print(e)
    return state
```

### 不要在节点内重新排序 `interrupt` 调用

LangGraph 维护特定于执行任务的恢复值列表。匹配严格基于索引，所以节点内中断调用的顺序很重要。

✅ 好的做法 - 保持中断调用顺序一致：

```python
def node_a(state: State):
    # ✅ Good: interrupt calls happen in the same order every time
    name = interrupt("What's your name?")
    age = interrupt("What's your age?")
    city = interrupt("What's your city?")

    return {
        "name": name,
        "age": age,
        "city": city
    }
```

❌ 不好的做法 - 有条件地跳过中断：

```python
def node_a(state: State):
    # ❌ Bad: conditionally skipping interrupts changes the order
    name = interrupt("What's your name?")

    # On first run, this might skip the interrupt
    # On resume, it might not skip it - causing index mismatch
    if state.get("needs_age"):
        age = interrupt("What's your age?")

    city = interrupt("What's your city?")

    return {"name": name, "city": city}
```

### 不要在 `interrupt` 调用中返回复杂值

复杂值可能无法序列化。最佳实践是仅使用可以被合理序列化的值。

✅ 好的做法 - 传递简单的、可 JSON 序列化的类型：

```python
def node_a(state: State):
    # ✅ Good: passing simple types that are serializable
    name = interrupt("What's your name?")
    count = interrupt(42)
    approved = interrupt(True)

    return {"name": name, "count": count, "approved": approved}
```

✅ 好的做法 - 传递具有简单值的字典/对象：

```python
def node_a(state: State):
    # ✅ Good: passing structured data with simple values
    response = interrupt({
        "question": "What's your name?",
        "required": True,
        "options": ["Alice", "Bob", "Charlie"]
    })
    return {"name": response}
```

❌ 不好的做法 - 传递函数或类实例：

```python
def validate_input(value):
    return len(value) > 0

def node_a(state: State):
    # ❌ Bad: passing a function to interrupt
    # The function cannot be serialized
    response = interrupt({
        "question": "What's your name?",
        "validator": validate_input  # This will fail
    })
    return {"name": response}
```

### 在 `interrupt` 之前调用的副作用必须是幂等的

由于中断的工作方式是重新运行调用它们的节点，`interrupt` 之前调用的副作用应该是幂等的。幂等意味着可以多次应用相同的操作，而不会改变初始执行以外的结果。

✅ 好的做法 - 使用幂等操作（upsert）：

```python
def node_a(state: State):
    # ✅ Good: using upsert operation which is idempotent
    # Running this multiple times will have the same result
    db.upsert_user(
        user_id=state["user_id"],
        status="pending_approval"
    )

    approved = interrupt("Approve this change?")

    return {"approved": approved}
```

✅ 好的做法 - 将副作用放在 `interrupt` 调用之后：

```python
def node_a(state: State):
    approved = interrupt("Approve this change?")
    
    if approved:
        # Side effect after interrupt
        db.create_audit_log({
            "user_id": state["user_id"],
            "action": "approved"
        })
    
    return {"approved": approved}
```

✅ 好的做法 - 分离到不同节点：

```python
def interrupt_node(state: State):
    return {"approved": interrupt("Approve this change?")}

def effect_node(state: State):
    if state["approved"]:
        db.create_audit_log({
            "user_id": state["user_id"],
            "action": "approved"
        })
    return state
```

❌ 不好的做法 - 在 `interrupt` 之前创建新记录：

```python
def node_a(state: State):
    # ❌ Bad: creating a new record before interrupt
    # This will create duplicate records on each resume
    audit_id = db.create_audit_log({
        "user_id": state["user_id"],
        "action": "pending_approval",
        "timestamp": datetime.now()
    })

    approved = interrupt("Approve this change?")

    return {"approved": approved, "audit_id": audit_id}
```

## 对作为函数调用的子图使用中断

在节点内调用子图时，父图将从调用子图且触发 `interrupt` 的节点开头恢复执行。类似地，子图也将从调用 `interrupt` 的节点开头恢复。

```python
def node_in_parent_graph(state: State):
    some_code()  # <-- This will re-execute when resumed
    # Invoke a subgraph as a function.
    # The subgraph contains an `interrupt` call.
    subgraph_result = subgraph.invoke(some_input)
    # ...

def node_in_subgraph(state: State):
    some_other_code()  # <-- This will also re-execute when resumed
    result = interrupt("What's your name?")
    # ...
```

## 使用中断进行调试

要调试和测试图，可以将静态中断用作断点，一次一个节点地逐步执行图。静态中断在节点执行之前或之后的定义点触发。

### 在编译时设置断点

```python
graph = builder.compile(
    interrupt_before=["node_a"],
    interrupt_after=["node_b", "node_c"],
    checkpointer=checkpointer,
)

# Pass a thread ID to the graph
config = {
    "configurable": {
        "thread_id": "some_thread"
    }
}

# Run the graph until the breakpoint
graph.invoke(inputs, config=config)

# Resume the graph
graph.invoke(None, config=config)
```

### 在运行时设置断点

```python
config = {
    "configurable": {
        "thread_id": "some_thread"
    }
}

# Run the graph until the breakpoint
graph.invoke(
    inputs,
    interrupt_before=["node_a"],
    interrupt_after=["node_b", "node_c"],
    config=config,
)

# Resume the graph
graph.invoke(None, config=config)
```

## 使用 LangSmith Studio

可以使用 LangSmith Studio 在 UI 中运行图之前设置静态中断。还可以使用 UI 在执行的任何点检查图状态。

---
来源：https://docs.langchain.org.cn/oss/python/langgraph/interrupts
