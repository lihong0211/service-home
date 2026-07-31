# 内存 (Memory overview)

## 短期记忆

**短期记忆**让应用程序能够"记住单个线程或对话中的先前交互"。LangGraph 将其作为代理状态的一部分进行管理，通过线程范围的检查点持久化。状态通常包括对话历史记录以及其他有状态数据，例如上传的文件、检索的文档或生成的工件。

### 管理短期记忆

对话历史记录是最常见的短期记忆形式。完整历史记录可能无法适应 LLM 的上下文窗口。由于上下文窗口有限且消息列表成本很高，许多应用程序可以受益于使用技术来手动删除或忘记陈旧信息。

## 长期记忆

**长期记忆**在 LangGraph 中允许"系统在不同的对话或会话中保留信息"。与仅限于线程范围的短期记忆不同，长期记忆保存在自定义"命名空间"中。

长期记忆没有一种通用的解决方案。关键问题包括：

1. **记忆的类型是什么？** 人类使用记忆来记住事实、经历和规则。AI 代理可以采用相同方式。

2. **何时更新记忆？** 记忆可以作为代理应用程序逻辑的一部分进行更新（"在热路径上"），或作为后台任务进行更新。

### 记忆类型对照表

| 记忆类型 | 存储的内容 | 人类示例 | 代理示例 |
|---------|---------|--------|--------|
| 语义记忆 | 事实 | 在学校学到的东西 | 关于用户的事实 |
| 情景记忆 | 经历 | 做过的事情 | 过去的代理操作 |
| 程序记忆 | 指令 | 本能或运动技能 | 代理系统提示 |

## 语义记忆

**语义记忆**涉及保留特定的事实和概念。对于 AI 代理，语义记忆通常用于通过记住过去交互中的事实或概念来个性化应用程序。

### 档案

语义记忆可以通过不同方式管理。例如，记忆可以是单个、不断更新的"概要"，其中包含关于用户、组织或其他实体的明确范围和特定信息（包括 JSON 文档）。

在记住概要时，需要确保每次都**更新**概要。因此，需要传入先前的概要并要求模型生成新的概要（或应用于旧概要的某些 JSON 补丁）。随着概要变得越来越大，这可能会变得容易出错，并可能受益于将概要拆分为多个文档或在生成文档时使用**严格**解码。

### 集合

或者，记忆可以是不断更新和扩展的文档集合。每个单独的记忆可以范围更窄且更易于生成，这意味着不太可能随着时间的推移而丢失信息。LLM 产生新信息比协调新信息与现有概要更容易。因此，文档集合往往会导致下游更高的召回率。

但这会将一些复杂性转移到记忆更新上。模型现在必须删除或更新列表中的现有项目，这可能很棘手。此外，一些模型可能会默认过度插入，而另一些模型可能会默认过度更新。

## 情景记忆

**情景记忆**涉及回忆过去的事件或行动。CoALA 论文阐述：事实可以写入语义记忆，而**经历**可以写入情景记忆。对于 AI 代理，情景记忆通常用于帮助代理记住如何完成一项任务。

在实践中，情景记忆通常通过少样本示例提示来实现，代理从过去的序列中学习以正确执行任务。有时"展示"比"讲述"更容易，大型语言模型（LLM）也能很好地从示例中学习。

## 程序记忆

**程序记忆**涉及记住执行任务所用的规则。在人类中，程序记忆就像执行任务的内在知识。对于 AI 代理，程序记忆是模型权重、代理代码和代理提示的组合。

一种有效的改进代理指令的方法是通过"反思"或元提示。这涉及使用其当前指令（例如，系统提示）以及最近的对话或明确的用户反馈来提示代理。然后，代理根据此输入完善其自身的指令。

### 程序记忆实现示例

```python
# Node that *uses* the instructions
def call_model(state: State, store: BaseStore):
    namespace = ("agent_instructions", )
    instructions = store.get(namespace, key="agent_a")[0]
    # Application logic
    prompt = prompt_template.format(instructions=instructions.value["instructions"])
    ...

# Node that updates instructions
def update_instructions(state: State, store: BaseStore):
    namespace = ("instructions",)
    instructions = store.search(namespace)[0]
    # Memory logic
    prompt = prompt_template.format(instructions=instructions.value["instructions"], conversation=state["messages"])
    output = llm.invoke(prompt)
    new_instructions = output['new_instructions']
    store.put(("agent_instructions",), "agent_a", {"instructions": new_instructions})
    ...
```

## 写入记忆

代理编写记忆有两种主要方法：**"在热路径中"**和**"在后台"**。

### 在热路径中

在运行时创建记忆既有优点也有挑战。

**优点：**
- 允许实时更新，使新的记忆立即可用于后续交互
- 能够实现透明度，因为用户可以在创建和存储记忆时收到通知

**挑战：**
- 如果代理需要新的工具来决定提交到记忆的内容，则可能会增加复杂性
- 推理保存到记忆的内容的过程可能会影响代理的延迟
- 代理必须在记忆创建和其他职责之间执行多任务处理

例如，ChatGPT 使用 save_memories 工具以内容字符串的形式插入记忆。

### 在后台

作为单独的后台任务创建记忆提供了几个优点：
- 消除了主应用程序中的延迟
- 将应用程序逻辑与记忆管理分离
- 允许代理更专注于完成任务
- 提供在时间上创建记忆的灵活性，以避免重复工作

**挑战：**
- 确定记忆写入的频率变得至关重要，因为不频繁的更新可能会导致其他线程没有新的上下文
- 确定何时触发记忆形成也很重要
- 常见的策略包括在设置的时间段后安排、使用 cron 计划或手动触发

## 记忆存储

LangGraph 将长期记忆作为 JSON 文档存储在存储中。每个记忆都组织在一个自定义的 `namespace`（类似于文件夹）和一个不同的 `key`（如文件名）下。命名空间通常包括用户或组织 ID 或其他便于组织信息的标签。

### 记忆存储实现示例

```python
from langgraph.store.memory import InMemoryStore


def embed(texts: list[str]) -> list[list[float]]:
    # Replace with an actual embedding function or LangChain embeddings object
    return [[1.0, 2.0] * len(texts)]


# InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production use.
store = InMemoryStore(index={"embed": embed, "dims": 2})
user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_context)
store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "my-key": "my-value",
    },
)
# get the "memory" by ID
item = store.get(namespace, "a-memory")
# search for "memories" within this namespace, filtering on content equivalence, sorted by vector similarity
items = store.search(
    namespace, filter={"my-key": "my-value"}, query="language preferences"
)
```

---
来源：https://docs.langchain.org.cn/oss/python/langgraph/memory
