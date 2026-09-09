# 模型微调模块（Finetuning）

> 代码位置：`service/ai/finetuning/`
> 覆盖文件：`finetuning.py`、`legal.py`、`legal_5090.py`、`medical.py`、`medical_5090.py`、`eval_compare_lora.py`、`paths.py`
> 生成日期：2026-08-18（全量重写）

---

## 第一部分：背景演进

### 问题背景

通用基座模型（Qwen2.5 系列等）在开放域对话上表现良好，但在医疗问诊、法律咨询这类专业场景中，输出往往术语不准确、格式不稳定、缺少领域内应有的"免责—建议—转诊"结构。让通用模型在专业场景可用，通常有三条路：加长 Prompt（成本随对话轮次线性增长，且无法让模型真正"记住"领域知识）、检索增强 RAG（依赖知识库质量，无法改变模型的表达风格与推理习惯）、微调（Finetuning，直接在领域数据上继续训练模型参数）。本模块选择的是第三条路。

**全参数微调的局限**：对 1.5B～7B 规模的模型做全参数微调，需要为模型权重、优化器状态（Adam 需要额外 2 份与参数等大的动量缓存）、梯度分别保留显存，7B 模型全量 fp32 微调通常需要 80GB+ 显存，消费级 GPU（如 24～32GB 显存的 RTX 4090/5090）无法承受；即便硬件够用，全参数微调也容易发生"灾难性遗忘"——模型在拟合领域数据的同时丢失原有的通用对话能力。

### 核心概念

- **LoRA（Low-Rank Adaptation，低秩适配）**：冻结基座模型的全部原始权重，只在 Attention 与 MLP 的关键投影层（`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`）旁插入一对低秩矩阵 `A(d×r)、B(r×d)` 参与训练，前向时输出叠加 `B·A·x`。可训练参数量降到原模型的 0.1%～1%，显存占用主要由基座权重（推理精度）和极小的 LoRA 参数决定，不再需要为全量参数保存优化器状态。
- **4bit 量化（QLoRA 思路）**：借助 `bitsandbytes` 的 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)` 将基座权重压缩到 4bit 存储、反量化到 bf16/fp16 参与计算，配合 `prepare_model_for_kbit_training` 对齐 LayerNorm 精度并开启梯度检查点。本模块中 `legal.py`、`medical.py` 支持按需开启（`USE_4BIT=1`），`_5090.py` 系列因显存充裕（RTX 5090 = 32GB）默认关闭，直接全精度 bf16 训练。
- **Unsloth 加速框架**：通过自定义 Triton 内核、算子融合与显存复用重写 LoRA 前反向计算路径，官方宣称训练速度提升 2～5 倍、显存降低最多 60%，接口上用 `FastLanguageModel.from_pretrained` / `FastLanguageModel.get_peft_model` 替代原生 `AutoModelForCausalLM` + `peft.get_peft_model`，仅限 CUDA 设备。

### 演进脉络

| 阶段 | 方案 | 特点 |
|---|---|---|
| 早期 | 全参数微调 | 效果上限高，但显存/存储成本随模型规模线性增长，7B 起步即需多卡 |
| 2021 | LoRA（微软论文） | 冻结基座，训练旁路低秩矩阵，参数量降至 <1% |
| 2023 | QLoRA | LoRA + 4bit 量化基座，消费级单卡可训练 7B+ 模型 |
| 2024 前后 | Unsloth 等训练加速框架 | 工程层面重写内核，同等硬件下吞吐提升数倍 |
| 当前 | 1.5B 级小模型 + 高质量领域数据 | 追求本地可部署、低延迟，用更精的数据弥补参数量劣势 |

### 本模块定位

`service/ai/finetuning/` 内的文件分为三类，彼此不共享运行时进程：

1. **训练脚本**（`legal.py`、`legal_5090.py`、`medical.py`、`medical_5090.py`）：命令行直接运行（`python -m service.ai.finetuning.xxx`），不暴露 HTTP 接口，训练产物（LoRA adapter）落盘到 `<project_root>/lora/` 下。
2. **推理服务**（`finetuning.py`）：常驻主 FastAPI 进程，通过 `POST /ai/finetuning/chat` 加载训练好的 LoRA adapter 提供问答；同时导出 `GET /ai/finetuning/lora-options` 供前端拉取可选模型列表。
3. **评测脚本**（`eval_compare_lora.py`）：独立命令行工具，用训练数据前 N 条抽样跑一遍 LoRA 推理，产出"标准答案 vs 模型生成"的人工核查用 Markdown。

三者通过 `paths.py` 定义的目录约定串联：训练脚本写入、推理服务读取、评测脚本可读取同一批目录，是本模块唯一的耦合点。

---

## 第二部分：架构剖析

### 整体分层

```
训练层（离线，命令行）          推理层（在线，HTTP）           评测层（离线，命令行）
─────────────────────         ─────────────────────         ─────────────────────
medical_5090.py  ─┐                                          eval_compare_lora.py
legal_5090.py    ─┼─ 写入 → lora/{日期}_{模型名}/ ← 读取 ─ finetuning.py ← 读取 ─┘
medical.py       ─┤            (adapter_model.safetensors)   (POST /ai/finetuning/chat)
legal.py（已损坏）─┘
                        ↑
                  paths.py 统一定义目录规则
```

### 普通版与 `_5090` 版本的关系

代码上看，两者是同一目标（产出医疗/法律 LoRA）的两条独立实现，而不是"新旧迭代替换"关系：

| 维度 | `medical_5090.py` / `legal_5090.py` | `medical.py` / `legal.py` |
|---|---|---|
| 训练框架 | `unsloth.FastLanguageModel` | 原生 `transformers.AutoModelForCausalLM` + `peft.LoraConfig` |
| 设备要求 | 仅 CUDA（依赖 Flash Attention） | CUDA / MPS（Apple Silicon）/ CPU 均可运行 |
| 训练数据 | `dataset/【数据集】medical_hq/`（HuatuoGPT-SFT 22.6 万条 + 百科 36.2 万条）、`dataset/【数据集】legal_hq/DISC-Law-SFT/`（Pair-QA 7.97 万条 + Triplet-QA 2.33 万条） | `medical.py` 用本地"【数据集】中文医疗数据"CSV（男科/内科/外科/儿科/肿瘤科/妇产科 6 类）；`legal.py` 引用 `service.ai.finetuning.dataset_legal.load_legal_data`，但该模块**源文件已不存在于仓库**（`service/ai/finetuning/__pycache__/dataset_legal.cpython-312.pyc` 是仅存痕迹），`legal.py` 目前**无法运行** |
| 样本格式 | `tokenizer.apply_chat_template`（system + user + assistant 三段式），训练时注入领域 System Prompt | 旧式 `### 问题：\n{}\n\n### 回答：\n{}` 纯文本拼接，不使用 chat template，也不含 system 段 |
| LoRA 超参 | `r=16, lora_alpha=32, lora_dropout=0.05` | `r=16, lora_alpha=16, lora_dropout=0` |
| 量化 | 不量化，全精度 bf16（32GB 显存足够） | 可选 4bit（`USE_4BIT=1`），CUDA/MPS/CPU 自适应精度 |

**关键事实（已用 `adapter_config.json` 核实）**：当前部署在 `lora/medical/`、`lora/legal/` 下、被 `finetuning.py` 实际加载用于线上推理的两个 adapter，其 `lora_alpha=32`、含 `chat_template.jinja` 文件，与 `_5090.py` 系列的训练超参完全吻合，而非普通版（`alpha=16`、无 chat template）的产物。也就是说：**普通版 `medical.py`/`legal.py` 更接近本地调试/历史遗留脚本，并未产出当前线上使用的 LoRA 权重**；`legal.py` 更因缺失依赖模块处于不可运行状态。这一点在下文"设计决策与取舍"和"已知局限"中会展开为具体风险点。

### 核心数据流：一次训练到模型产出的完整路径

```
1. 原始数据（CSV / JSONL）
   ├─ medical.py:      本地 CSV（6 科室），read_csv_with_encoding 自动探测 gbk/gb2312/gb18030/utf-8
   ├─ medical_5090.py: HuatuoGPT_sft_data_v1.jsonl（"问：/答："多轮取首轮）+ 百科 train_datasets.jsonl
   ├─ legal_5090.py:   DISC-Law-SFT-Pair-QA / Triplet-QA .jsonl（input/output 字段）
   └─ legal.py:        （数据模块缺失，此路径已断）

2. 清洗与过滤
   过滤过短/过长样本（如 medical_5090 的 load_huatuo_sft 丢弃 q<5 或 a<10 字符，
   q>600 或 a>800 字符的样本），统一转为 {"question", "answer"} 或 {"input","output"} 字典列表

3. 格式化为训练文本
   ├─ 旧式模板：f"### 问题：\n{q}\n\n### 回答：\n{a}" + EOS_TOKEN（medical.py / legal.py）
   └─ chat template：tokenizer.apply_chat_template([system, user, assistant])（_5090.py 系列）

4. 加载基座 + 注入 LoRA
   AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=..., [quantization_config])
   → [prepare_model_for_kbit_training if 4bit]
   → get_peft_model(model, LoraConfig(...))  或  FastLanguageModel.get_peft_model(...)

5. SFTTrainer 训练
   trl.SFTTrainer(model, processing_class=tokenizer, train_dataset=dataset, args=SFTConfig(...))
   trainer.train()

6. 产物落盘
   model.save_pretrained(LORA_SAVE_DIR)       # adapter_config.json + adapter_model.safetensors
   tokenizer.save_pretrained(LORA_SAVE_DIR)   # tokenizer.json / chat_template.jinja 等
   LORA_SAVE_DIR = <project_root>/lora/{日期}_{模型名}/   （paths.get_run_parent_dir 计算）

7. 推理服务加载
   finetuning.get_model(lora_type) 懒加载 base + PeftModel.from_pretrained(base, lora_path)
   → tokenizer.apply_chat_template / 旧模板 拼 prompt → model.generate → 流式或整体返回

8.（可选）回归评测
   eval_compare_lora.py 从同一批原始数据取前 N 条，用相同 chat_template + 超参重新推理，
   与训练集标准答案并排写入 Markdown，供人工核查微调效果
```

### 关键设计原则

- **训练与推理解耦**：训练脚本是一次性命令行任务，推理服务是常驻进程，两者仅通过磁盘上的 `lora/` 目录通信，互不阻塞、互不共享 Python 进程状态。
- **路径规则单一来源**：所有目录拼接逻辑集中在 `paths.py`（`get_run_parent_dir`/`get_lora_dir`/`get_outputs_hf_dir`/`get_latest_lora_dir`），训练脚本与推理服务、评测脚本共用同一套函数，避免路径硬编码分散在各文件。
- **懒加载优先**：推理服务不在应用启动时加载任何模型，首次收到某个 `lora_type` 的请求时才加载（见 `finetuning.get_model`），避免拖慢主服务启动、避免为不会被用到的领域模型常驻占用显存。
- **Base vs LoRA 双路对比**：推理服务默认非流式接口会同时跑一次纯基座（`get_base_model`）和一次基座+LoRA（`get_model`），把两段回复都返给调用方，便于直观感受微调收益，而不是只呈现单一结果。

### 与行业标准方案对比

| 维度 | Unsloth + LoRA（`_5090.py`） | 全参数微调 | HF 原生 LoRA（`medical.py`/`legal.py`） | QLoRA（4bit + LoRA） |
|---|---|---|---|---|
| 可训练参数占比 | ~0.1%～1%（LoRA 层） | 100% | ~0.1%～1%（LoRA 层） | ~0.1%～1%（LoRA 层） |
| 1.5B 模型显存需求 | 全精度 bf16，约 8～10GB 起（本模块 batch=32/48 峰值约 25GB） | fp32 全参 + 优化器状态，数十 GB | 4bit 量化时约 4～6GB，全精度约 8～10GB | 与 HF 原生 4bit 一致，约 4～6GB |
| 训练速度 | 最快（Triton 融合内核，官方号称 2～5x） | 基准最慢（反向传播覆盖全部参数） | 中等（标准 PyTorch/HF 实现） | 略慢于全精度 LoRA（量化/反量化开销） |
| 设备兼容性 | 仅 CUDA（依赖 Flash Attention） | CUDA / 多卡 | CUDA / MPS / CPU | 仅 CUDA（`bitsandbytes` 限制） |
| 精度/效果上限 | 与 LoRA 理论一致，工程加速不改变收敛效果 | 理论上限最高 | 与 Unsloth 版一致（同为 LoRA） | 4bit 量化带来轻微精度损失，通常可接受 |
| 适用场景 | 有独立 GPU（如 5090/云卡）、追求训练速度 | 资源充裕、需要最大化效果、可接受高成本 | 本地 Mac / 低显存机器、快速调试 | 显存紧张但仍需 CUDA 的场景 |

**选型建议**：有独立 CUDA 显卡（本地 5090 或云 GPU 实例）且追求训练效率时，用 `_5090.py` 系列（Unsloth）；仅有 Mac 或显存紧张的 CUDA 卡做本地调试验证时，用普通版 + `USE_4BIT=1`；若显存严重不足（如 8GB 以下）且必须用 CUDA，QLoRA（普通版开 4bit）是唯一可行路径；全参数微调在本模块当前规模（1.5B）下性价比不高，除非有明确证据表明 LoRA 表达能力不足。

---

## 第三部分：代码实现深度解析

### 核心函数/类清单

| 函数/类 | 所在文件 | 作用 |
|---|---|---|
| `get_run_parent_dir()` / `get_latest_lora_dir()` | `paths.py` | 统一计算/查找 `lora/{日期}_{模型名}/` 目录，供训练与推理共用 |
| `get_model(lora_type)` / `get_base_model(lora_type)` | `finetuning.py` | 懒加载单例，按 `lora_type` 缓存 (model, tokenizer)，线程锁防并发重复加载 |
| `_prompt_for_sft_lora(lora_type, messages, tokenizer)` | `finetuning.py` | 按领域还原训练时的 prompt 格式（legal 走 chat template，medical 走旧模板） |
| `chat_stream_compare(messages, options, lora_type)` | `finetuning.py` | 双线程并行流式生成 base 与 base+LoRA 两路输出，用共享 `Queue` 按到达顺序汇聚 |
| `_fix_tokenizer_dict_bug()` / `_fix_sft_trainer_eos_check()` | `legal_5090.py` / `medical_5090.py` | 启动期直接改写已安装的 `transformers`/`trl` 源码文件，修复版本兼容 bug |
| `load_model_and_tokenizer()` / `generate()` | `eval_compare_lora.py` | 评测脚本加载 base+LoRA 并按 chat template 单条推理 |
| `formatting_prompts_func()` / `format_sample()` | 四个训练脚本 | 数据集到训练文本的格式化函数（旧模板 vs chat template 两种实现） |

### 关键实现细节

**4bit 量化路径**（`legal.py` / `medical.py`）：
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, quantization_config=bnb_config, device_map="auto", ...)
model = prepare_model_for_kbit_training(model)
```
`medical.py` 在检测到 CUDA 时默认 `use_4bit=True`（不留开关，硬编码开启）；`legal_5090.py`/`medical_5090.py` 走 Unsloth 自有加载路径，`load_in_4bit=False` 固定关闭量化；`legal.py`（普通版）通过环境变量 `USE_4BIT=1` 手动开启，默认走全精度分支。三处 4bit 开关的触发条件并不统一，是阅读代码时容易踩的细节。

**GRPO / R1 强化学习式对齐**：当前 `service/ai/finetuning/` 目录内**没有**任何 GRPO（Group Relative Policy Optimization）或 R1 风格推理链训练的实现——全部四个训练脚本均为标准 SFT（`trl.SFTTrainer` + 交叉熵损失），不涉及奖励模型、策略采样或 KL 约束。旧版本文档（2026-03-11）的变更记录提到历史上曾有 Qwen2.5-7B + GRPO/R1 相关脚本，但在当前代码库中已被移除，本轮通读代码未发现任何残留痕迹。若未来需要偏好对齐或推理链能力，需要新增 `GRPOTrainer`（TRL 已内置）或引入 DPO/ORPO 数据管线，属于全新工作量，而非在现有脚本上开关切换。

**评测对比逻辑**（`eval_compare_lora.py`）：`load_medical_samples(n)` / `load_legal_samples(n)` 分别从训练所用的原始 JSONL 中按相同的过滤规则（长度阈值）重新解析出前 N 条 `{question, answer}`，不依赖训练时缓存的 `datasets.Dataset` 对象（两者是完全独立的两次文件解析）。`generate()` 用与 `_5090.py` 训练一致的 `system + user` chat template 拼 prompt，`max_new_tokens=1000`（远高于训练脚本推理示例里的 256），避免专业领域长回答被截断而造成评测失真。最终结果写入 `finetuning/eval_compare_{type}_{时间戳}.md`，人工比对"标准答案"与"LoRA 生成"两栏。

### 设计决策与取舍

**决策 1：训练脚本按框架拆成独立文件，而非用参数开关切换 Unsloth/HF 原生**
Unsloth 的 `FastLanguageModel` 与原生 `AutoModelForCausalLM + peft.LoraConfig` 在模型加载、tokenizer 处理、`get_peft_model` 调用方式上均不同，且 Unsloth 只支持 CUDA。用独立文件避免脚本内充斥 `if HAS_UNSLOTH` 分支判断，代价是同一份数据清洗逻辑（如医疗 CSV 编码探测、法律 JSONL 长度过滤）在多个文件里重复实现，后续新增领域时需要同步改动多处。

**决策 2：对已安装的第三方库源码做启动期猴子补丁**
`legal_5090.py`/`medical_5090.py` 的 `_fix_tokenizer_dict_bug()` 直接用文本替换改写 `transformers/tokenization_utils_base.py`（修复 `_config.model_type` 在 `_config` 为 `dict` 时的 `AttributeError`），`_fix_sft_trainer_eos_check()` 用括号计数定位并替换 `trl/trainer/sft_trainer.py` 里的 `raise ValueError(...)` 块为 `pass`（跳过 `eos_token` 词表校验，因为 Unsloth 给 Qwen2 设置的 `<|im_end|>` 在旧词表校验逻辑下会被误判为不存在）。这是版本兼容问题的最小侵入方案，不需要锁定/降级 `transformers`、`trl` 版本；代价是直接修改 `site-packages` 下的文件，一旦升级依赖版本，补丁的字符串匹配（`old`/`marker`）可能失效且不会报错提醒，属于隐性风险。

**决策 3：推理服务用"全局字典 + 单把互斥锁"做懒加载缓存**
```python
_model_tokenizer = {}  # lora_type -> (model, tokenizer)
def get_model(lora_type="medical"):
    with _model_tokenizer_lock:
        if lora_type in _model_tokenizer:
            return _model_tokenizer[lora_type]
        ...  # 加载 base + LoRA，约需数秒
        _model_tokenizer[lora_type] = (model, tokenizer)
```
模型加载有明显耗时和显存/内存开销，不适合在应用启动时同步加载所有领域模型；用一把全局锁保证同一 `lora_type` 不会被并发请求重复加载。代价是锁粒度是全局而非按 `lora_type` 分离——如果 `medical` 和 `legal` 两个从未加载过的类型同时收到首个请求，会互相排队等待，而不是并行加载。

**决策 4：`medical` 类型的推理 prompt 格式与实际部署 adapter 的训练格式不一致（技术债，非有意设计）**
`finetuning.py::_prompt_for_sft_lora` 对 `lora_type=="medical"` 固定走旧式 `### 问题：\n{}\n\n### 回答：\n{}` 模板（对齐 `medical.py`），但通过 `adapter_config.json` 核实，当前 `lora/medical/` 目录下的权重 `lora_alpha=32` 且带 `chat_template.jinja`，与 `medical_5090.py`（chat template + System Prompt）的训练配置一致，而非 `medical.py`（`alpha=16`、旧模板）。也就是说线上实际推理时构造的 prompt 格式，很可能与该 adapter 训练时见过的格式不同——LoRA 对训练分布外的输入格式泛化能力有限，这是一个需要重点核实、可能已经在悄悄拉低回答质量的隐患，而非刻意的架构选择。

**决策 5：`legal` 类型的基座声明与实际 adapter 训练基座不一致**
`finetuning.py` 中 `LEGAL_BASE_MODEL_NAME = "Qwen2.5-1.5B"`（无 `-Instruct` 后缀的 Base 基座），源码注释写明"法律 LoRA 在 Base 基座上训练，推理时也用 Base"。但 `lora/legal/adapter_config.json` 的 `base_model_name_or_path` 字段记录为 `.../models/Qwen/Qwen2.5-1.5B-Instruct`，与 `legal_5090.py`（用 Instruct 基座）的训练配置吻合，而非 `legal.py`（`USE_BASE_MODEL=1` 时才用 Base）产物。Base 与 Instruct 版本架构相同、张量形状一致，`PeftModel.from_pretrained` 不会报错，但两者预训练阶段不同（Instruct 经过指令对齐），拼接错误的基座大概率影响生成质量而不会有显式异常提示。

**决策 6：`LORA_TYPE_DIRS["legal"]` 硬编码带日期的目录名**
```python
LORA_TYPE_DIRS = {
    "medical": "medical",
    "legal": "20260309_Qwen2.5-1.5B-legal-5090",
}
```
`medical` 用无日期的约定目录名 `lora/medical`，`legal` 却硬编码了一个带训练日期的目录名。本地检查 `lora/` 目录下实际只有 `lora/legal`、`lora/medical` 两个无日期目录，`lora/20260309_Qwen2.5-1.5B-legal-5090` 并不存在——若无 `FINETUNING_LORA_PATH` 环境变量兜底，本地环境下 `lora_type=legal` 的请求会直接抛 `FileNotFoundError`。这段硬编码大概率是为线上/训练服务器环境准备的路径，与当前 dev 分支的目录状态脱节，属于"训练完成后未同步更新推理配置"的典型协作断层。

**决策 7：LoRA `alpha/r` 比例在两套脚本中不同**
```python
r=16, lora_alpha=32   # _5090.py 系列，2:1
r=16, lora_alpha=16   # 普通版，1:1
```
`lora_alpha/r` 决定 LoRA 输出的缩放系数，2:1 相当于放大 LoRA 分支对最终输出的影响权重。`_5090.py` 系列数据量更大（十万级）、System Prompt 更复杂，用更高缩放系数强化学习信号；普通版数据量小、本地调试场景，用保守的 1:1 降低过拟合风险。这是刻意的超参选择，与决策 4/5 描述的"非预期不一致"性质不同。

---

## 第四部分：应用场景与实战

### 核心使用场景

1. **医疗问诊问答**：前端/调用方通过 `POST /ai/finetuning/chat` 传 `lora_type=medical`，获取基座+医疗 LoRA 的回复（研究/演示定位，非正式医疗建议，医疗 System Prompt 内已声明"不做确定性诊断"）。
2. **法律咨询问答**：同接口切换 `lora_type=legal`，回复中会尝试引用《民法典》《劳动合同法》等具体法条并给出处理流程。
3. **对比展示微调收益**：非流式接口默认返回 `base`（纯基座）与 `lora`（基座+LoRA）两段文本；流式接口在 `compare=true` 时通过 SSE 用 `source: "base"|"lora"` 区分两路 chunk，供前端并排展示。
4. **新领域/新数据的训练与验证**：参照 `_5090.py` 或普通版脚本结构新增领域，训练完成后跑 `eval_compare_lora.py` 做人工核查回归。

### 快速上手

**环境依赖**（`requirements.finetuning.txt`，建议单独装在训练/GPU 环境，不与主服务依赖混装）：
```
torch>=2.5.1
transformers>=4.51.3
datasets>=3.1.0
peft>=0.15.0
trl>=0.15.2
accelerate
bitsandbytes>=0.45.4
unsloth>=2025.3.19
pandas>=2.0.3
```
```bash
pip install -r requirements.finetuning.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
注：`unsloth` 仅在 CUDA 环境可用；Mac（MPS）/CPU 环境只需要 `torch/transformers/peft/trl/pandas`，`bitsandbytes` 装了也用不到（4bit 在非 CUDA 设备不可用）。

**示例 1：调用推理服务 HTTP 接口**
```python
import requests

# 医疗问诊，非流式，返回 base 与 lora 双路对比
res = requests.post("http://localhost:3000/ai/finetuning/chat", json={
    "messages": [{"role": "user", "content": "我最近总是感觉头晕，应该怎么办？"}],
    "lora_type": "medical",
})
data = res.json()
print("基座回复:", data["base"])
print("LoRA 回复:", data["lora"])

# 查询可选 LoRA 列表
opts = requests.get("http://localhost:3000/ai/finetuning/lora-options").json()
print(opts["data"])  # [{"id": "medical", ...}, {"id": "legal", ...}]
```

**示例 2：本地/GPU 服务器直接运行训练脚本**
```bash
cd <project_root>
# RTX 5090 上训练法律 LoRA（Unsloth，全量 DISC-Law-SFT 数据）
python -m service.ai.finetuning.legal_5090
# 完成后权重写入 lora/{当天日期}_Qwen2.5-1.5B-Instruct-legal-disc-5090/

# 本地 Mac / 低显存 CUDA 调试医疗 LoRA（HF 原生，可先把 MAX_TRAIN_SAMPLES 改小做冒烟测试）
python -m service.ai.finetuning.medical
```

**示例 3：训练完成后跑回归评测**
```bash
python -m service.ai.finetuning.eval_compare_lora --type legal --top 10
# 结果写入 service/ai/finetuning/eval_compare_legal_{时间戳}.md
# 若 LoRA 不在 lora/legal 约定目录下，用 --lora-dir 显式指定：
python -m service.ai.finetuning.eval_compare_lora --type legal --top 10 --lora-dir lora/20260309_Qwen2.5-1.5B-Instruct-legal-disc-5090
```

### 常见问题排查

- **`CUDA out of memory`**：`_5090.py` 系列全精度运行，1.5B 模型基础占用约 8～10GB，`per_device_train_batch_size=32/48` 时峰值可达 ~25GB；显存不够时优先调小 `per_device_train_batch_size` 并相应增大 `gradient_accumulation_steps` 维持等效 batch，或在 `legal_5090.py` 里改用普通版 + `USE_4BIT=1` 路线。
- **`AttributeError: 'dict' object has no attribute 'model_type'`**：`transformers` 某些版本在读取本地 tokenizer 配置时的已知 bug。`_5090.py` 系列脚本启动时会调用 `_fix_tokenizer_dict_bug()` 自动改写源码修复；`medical.py`/`finetuning.py` 则用 `PreTrainedTokenizerFast(tokenizer_file=...)` 手动构造 tokenizer 绕开该问题（见 `finetuning.py::_load_tokenizer`）。
- **`Ensure that the eos_token exists in the vocabulary`（SFTTrainer 报错）**：Unsloth 给 Qwen2 设置的 `<|im_end|>` eos token 与 `trl` 某些版本的词表校验逻辑冲突，`_5090.py` 脚本的 `_fix_sft_trainer_eos_check()` 会自动跳过该校验；手动排查时也可以 `tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})` 后重设 `tokenizer.eos_token_id`。
- **`POST /ai/finetuning/chat` 返回 404 且提示 LoRA 目录不存在**：检查 `finetuning.py::LORA_TYPE_DIRS["legal"]` 里硬编码的日期目录名是否与 `lora/` 下实际存在的目录一致；本地开发环境下该值默认指向一个不存在的目录（见"设计决策与取舍·决策 6"），可临时用环境变量 `FINETUNING_LORA_PATH=/绝对路径` 覆盖，或把该值改成本地实际的目录名。
- **`legal.py` 直接运行报 `ModuleNotFoundError: dataset_legal`**：该脚本依赖的 `service/ai/finetuning/dataset_legal.py` 已不在仓库中（只剩 `__pycache__` 里的旧 `.pyc`），当前不可运行；若需要 Base 基座 + 旧格式的法律训练路径，需要先补齐该数据加载模块，或改用已验证可运行的 `legal_5090.py`。
- **MPS（Apple Silicon）上梯度异常/NaN**：`gradient_checkpointing` 在 MPS 上与 meta device 存在冲突，`medical.py` 已按设备条件处理（`gradient_checkpointing=(device == "cuda")`），自行改造脚本时需保留这一判断。

---

## 第五部分：评估与展望

### 优势

- `_5090.py` 系列在 32GB 显存上可全精度 bf16 训练，无量化精度损失，同时受益于 Unsloth 的内核加速，是当前吞吐最高的训练路径。
- 推理服务（`finetuning.py`）与主 FastAPI 服务无缝集成，懒加载设计不拖慢应用启动，多领域模型按需加载、独立缓存。
- `paths.py` 把目录规则收敛成几个纯函数，训练脚本与推理/评测脚本共用同一套路径解析逻辑，避免路径拼接散落各处。
- 非流式接口默认返回 base/LoRA 双路结果、流式接口支持 `compare=true` 并行输出，直观呈现微调收益，便于业务方和评审快速判断效果。

### 局限与技术债务

- **训练格式与部署 adapter 不匹配的风险**（决策 4）：`finetuning.py` 对 `medical` 类型固定使用旧式 `###` 模板构造推理 prompt，而实际部署的 `lora/medical` 权重按 `adapter_config.json` 特征更像是 `medical_5090.py`（chat template + System Prompt）的产物。这一不一致目前未被文档或代码显式记录，属于最值得优先核实、可能已经影响线上回答质量的问题。
- **法律 LoRA 基座声明与实际不符**（决策 5）：源码注释与 `LEGAL_BASE_MODEL_NAME` 变量表明"legal 用 Base 基座"，但已部署 adapter 的 `base_model_name_or_path` 记录为 Instruct 基座，两者可能已经产生偏差。
- **`legal.py` 处于不可运行状态**：依赖的 `dataset_legal.py` 已从仓库中删除，是明确的死代码/技术债，长期保留会误导后来者以为这是可用的训练入口。
- **`LORA_TYPE_DIRS["legal"]` 路径硬编码含训练日期**（决策 6），本地开发环境与该硬编码值不一致，新一轮训练完成后需要手动改代码同步，容易遗漏。
- **两套训练脚本重复维护**：数据加载、格式化、训练参数在 `_5090.py` 与普通版之间各写一份，新增第三个领域时重复量还会增加。
- **推理服务全局锁**（决策 3）：不同 `lora_type` 首次并发请求时会互相等待，高并发冷启动场景下有排队延迟。

### 演进建议

- **短期**：核实并修复 `medical` 推理 prompt 格式与部署 adapter 训练格式的一致性（决策 4）；核实 `legal` 基座是否应统一为 Instruct，并同步修正源码注释与 `LEGAL_BASE_MODEL_NAME`；把 `LORA_TYPE_DIRS["legal"]` 改为调用 `get_latest_lora_dir()` 动态解析，消除日期硬编码；决定是否恢复或彻底删除 `legal.py`/`dataset_legal.py`。
- **中期**：抽取一套通用 LoRA 训练框架（数据加载接口 + 统一格式化 + 统一训练参数模板），各领域只需提供数据源和 System Prompt，减少 `_5090.py`/普通版之间的重复代码；把推理服务的全局锁换成按 `lora_type` 分片的锁，允许不同领域并行冷启动。
- **长期**：建设"训练产物 → 自动登记到推理服务"的最小流水线，训练脚本训练完成后自动更新推理侧的目录映射配置，避免手工同步遗漏；引入自动化的 base vs LoRA 对比评测（而非仅人工核查 Markdown），作为训练是否达标的量化门槛。

### 行业前沿

- **DoRA（权重分解 LoRA）**：将预训练权重分解为幅度与方向两部分，仅对方向部分做低秩适配，多数评测中效果优于标准 LoRA，Unsloth 已原生支持，属于本模块升级 LoRA 配置即可尝试的低成本改进。
- **偏好对齐（DPO/ORPO）**：在 SFT 之后（或与 SFT 合并为一步）引入偏好数据对齐，比 GRPO 等强化学习式方法工程复杂度更低，适合专业问答场景进一步压低"答非所问""语气不专业"等问题——这也是当前模块完全空白、若要补齐 GRPO/R1 能力时优先考虑的替代方案。
- **持续预训练（Continual Pretraining）+ SFT**：先在领域语料（医学文献、法律法规全文）上做无监督继续预训练，再叠加 SFT，比纯 SFT 更能让模型"理解"领域知识而非只记住问答对，是专业垂直模型的更完整方案，但对数据规模和算力要求显著更高。

---

## 变更记录

| 日期 | 变更说明 |
|---|---|
| 2026-08-18 全量重写 | 通读 `finetuning.py`、`legal.py`、`legal_5090.py`、`medical.py`、`medical_5090.py`、`eval_compare_lora.py`、`paths.py` 全部源码后重新撰写；新增对 `lora/medical`、`lora/legal` 实际 `adapter_config.json` 的核实，发现并记录"medical 推理 prompt 格式与部署 adapter 训练格式可能不一致""legal 基座声明与实际训练基座不符""`legal.py` 因缺失 `dataset_legal.py` 无法运行""`LORA_TYPE_DIRS['legal']` 硬编码路径与本地目录不符"等此前文档未涵盖的具体技术债；确认当前代码库中不存在 GRPO/R1 相关实现，不再沿用旧版文档中的相关表述。 |
