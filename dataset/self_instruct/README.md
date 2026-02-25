# Self-Instruct 数据生成

从少量**种子指令**出发，调用大模型批量生成新的任务指令，用于扩充 SFT / 指令微调数据。输出为 **Alpaca 风格**（instruction / input / output）的 JSONL，可直接或稍作转换后用于本仓库 `service/ai/14-Fine-tuning实操` 中的微调脚本。

## 目录结构

```
dataset/self_instruct/
├── README.md                 # 本说明
├── seed_tasks.jsonl          # 种子指令（可自行增删）
├── filters.py                # 指令过滤：长度、关键词、相似度等
├── self_instruct_pipeline.py # 主流水线：读种子 → 调用 LLM 生成 → 过滤 → 写 JSONL
├── run_self_instruct.py      # 一键运行脚本
├── output/                   # 默认输出目录（自动创建）
│   └── self_instruct_output.jsonl
└── __init__.py
```

## 环境

- Python 3.10+
- 安装依赖：`pip install openai`（用于调用 OpenAI 兼容 API，含 Qwen、本地部署等）

## 配置 API

任选其一即可：

- **Qwen**：在 [阿里云 Model Studio](https://www.alibabacloud.com/en) 创建 API Key，并设置环境变量：
  - `export QWEN_API_KEY=your_key`
  - 若使用阿里云提供的 OpenAI 兼容端点，可再设置 `OPENAI_BASE_URL`
- **OpenAI**：`export OPENAI_API_KEY=your_key`
- **本地 / 其他兼容服务**：`export OPENAI_BASE_URL=https://your-endpoint/v1` 和对应 `OPENAI_API_KEY` 或 `QWEN_API_KEY`

## 种子格式

`seed_tasks.jsonl` 每行一个 JSON，需包含 `instruction` 字段，例如：

```json
{"id": "seed_1", "instruction": "将给定的中文句子翻译成英文。"}
{"id": "seed_2", "instruction": "用一句话概括下面这段文字的主旨。"}
```

可自行增删、修改种子，以控制生成指令的风格和领域。

## 运行方式

### 方式一：项目根目录下以模块运行（推荐）

```bash
# 在项目根目录
export QWEN_API_KEY=your_key   # 或 OPENAI_API_KEY
python -m dataset.self_instruct.run_self_instruct
```

### 方式二：进入目录后直接运行

```bash
cd dataset/self_instruct
export QWEN_API_KEY=your_key
python run_self_instruct.py
```

### 可选环境变量

| 变量 | 含义 | 默认 |
|------|------|------|
| `NUM_INSTRUCTIONS` | 目标生成条数 | 20 |
| `SELF_INSTRUCT_MODEL` | 调用模型名 | gpt-3.5-turbo |
| `SELF_INSTRUCT_TEMPERATURE` | 采样温度 | 0.7 |

## 输出格式

默认输出到 `dataset/self_instruct/output/self_instruct_output.jsonl`，每行一条 Alpaca 风格数据，例如：

```json
{"instruction": "将下面要点改写成三条 bullet points。", "input": "", "output": ""}
```

`input`、`output` 为空，仅做占位；后续可对每条 `instruction` 再调用模型或人工补全 input/output，再用于微调。

## 过滤

流水线内置过滤（见 `filters.py`）：

- **LengthFilter**：指令长度在 [min_len, max_len]
- **KeywordFilter**：排除包含指定关键词（如 image / video）
- **PunctuationFilter**：排除以非字母数字开头
- **RougeSimilarityFilter**：与已有指令 n-gram 重叠过高则过滤

可在 `SelfInstructPipeline` 中传入自定义 `instruction_filter`（例如 `default_filter_config(...)` 或自行组合 `InstructionFilter`）。

## 与微调脚本对接

1. 用本流程生成 `output/self_instruct_output.jsonl`。
2. 为每条 `instruction` 补全 `input` / `output`（例如用同一 API 做一次“根据指令生成一条示例对话”）。
3. 将得到的 JSONL 转为 `service/ai/14-Fine-tuning实操` 所需格式（如 Alpaca 的 `instruction/input/output` 或医疗脚本的 question/answer），再运行对应微调脚本。

## 参考

- [CAMEL-AI Self-Instruct 文档](https://docs.camel-ai.org/cookbooks/data_generation/self_instruct_data_generation)
