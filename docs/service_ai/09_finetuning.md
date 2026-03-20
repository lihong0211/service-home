# 模型微调实验层（Finetuning）

> 文件：`service/ai/finetuning/`（medical_5090.py、legal_5090.py、medical.py、legal.py、finetuning.py、eval_compare_lora.py、paths.py、dataset_legal.py）  
> 生成日期：2026-03-11

---

## 第一部分：技术背景与演进

**问题背景**

通用大模型（GPT-4、Qwen-7B）在通用任务上表现优秀，但面对高度专业化场景（医疗诊断、法律条文、特定格式输出）时，回答往往不够精确、不符合行业术语规范，或需要大量 Prompt 工程才能勉强达标。微调（Finetuning）通过在领域数据上继续训练，让模型"内化"领域知识和输出风格，从根本上解决通用模型的领域适应问题。

**核心概念**

- **SFT（监督式微调）**：用"问题-答案"对训练模型学会特定格式和领域知识，是最常见的微调方式。
- **LoRA（低秩适配器）**：不改动原始权重，只在关键层旁插入低秩矩阵进行训练。1.5B 参数模型用 LoRA 微调只需更新约 0.1% 的参数，显存需求大幅降低。
- **Unsloth**：开源微调加速框架，通过 Flash Attention 2 + Triton 内核优化，使 SFT 速度提升 2-5 倍，显存占用减少约 60%。在 RTX 5090（32GB）上可全精度 bf16 训练 1.5B 模型，无需量化。

**演进脉络**

| 阶段 | 方案 | 特点 |
|------|------|------|
| 早期 | 全量参数微调（Full Fine-tuning） | 效果好，但需要数十 GB 显存，成本高 |
| 2021 | **LoRA**（微软） | 低秩矩阵，参数量极少，速度接近全量 |
| 2023 | QLoRA | LoRA + 4bit 量化，消费级 GPU 可微调 7B+ 模型 |
| 2024 | **Unsloth** | 工程加速，让 LoRA 训练快 2-5 倍 |
| 2025 | **1.5B 规模主流化** | 推理服务可本地部署，低延迟，与 7B 效果差距缩小 |

**本模块的定位**

`finetuning/` 目录包含两类内容：

1. **训练脚本**：在 GPU 机器上直接运行，不提供 HTTP 接口，生成的 LoRA 权重保存到 `<project>/lora/` 目录。
2. **推理服务**（`finetuning.py`）：集成到主服务，通过 `POST /ai/finetuning/chat` 对外提供 LoRA 微调模型的问答能力。

当前聚焦医疗与法律两个领域，覆盖「高质量新数据集（5090 版）」和「本地兼容版（HF 原生）」两套训练路线。

---

## 第二部分：架构剖析

**训练脚本双版本对比**

每个领域都有两个训练脚本：

| 脚本 | 适用环境 | 框架 | 特点 |
|------|---------|------|------|
| `medical_5090.py` | RTX 5090（32GB） | Unsloth + SFTTrainer | bf16 全精度，高 batch，速度最快 |
| `medical.py` | CUDA / MPS / CPU | HF 原生（transformers + peft） | 兼容性强，支持 4bit 量化，适合本地调试 |
| `legal_5090.py` | RTX 5090（32GB） | Unsloth + SFTTrainer | bf16 全精度，高 batch，速度最快 |
| `legal.py` | CUDA / MPS / CPU | HF 原生（transformers + peft） | 兼容性强，支持 4bit 量化，适合本地调试 |

**各训练脚本配置详情**

| 维度 | `medical_5090.py` | `legal_5090.py` | `medical.py` |
|------|------------------|----------------|-------------|
| 基座 | Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct | Qwen2.5-1.5B-Instruct |
| 数据集 | HuatuoGPT-SFT (226K) + 百科 (362K) | DISC-Law-SFT Pair-QA (79,692) + Triplet-QA (23,331) | 中文医疗数据 CSV（6 科室，默认取 10K） |
| LoRA r / alpha | 16 / 32 | 16 / 32 | 16 / 16 |
| batch size | 32 | 48 | 2–4（设备自适应） |
| epochs | 1 | 2 | 1–2 |
| 学习率 | 1e-4 | 5e-5 | 2e-4 |
| 量化 | 无（bf16 全精度） | 无（bf16 全精度） | 可选 4bit（CUDA 显存紧张时） |
| 输出目录 | `lora/日期_Qwen2.5-1.5B-Instruct-medical-huatuo-5090` | `lora/日期_Qwen2.5-1.5B-Instruct-legal-disc-5090` | `lora/日期_Qwen2.5-1.5B-Instruct` |

**lora 目录约定（paths.py）**

```
<project_root>/
└── lora/
    ├── 20260309_Qwen2.5-1.5B-Instruct-legal-disc-5090/   # 法律 LoRA
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── tokenizer.json
    │   └── outputs_hf/                                     # SFTTrainer checkpoint
    ├── medical/                                            # 医疗 LoRA（约定目录名，无日期）
    └── legal/                                              # 法律 LoRA（约定目录名，无日期）
```

`get_latest_lora_dir()` 按目录名（`{日期}_{模型名}` 格式）自动找最新一次训练结果。

**推理服务架构（finetuning.py）**

```
POST /ai/finetuning/chat
         │
         ▼
  get_model(lora_type)     ← 懒加载，线程安全，首次调用时加载 base + LoRA
         │
  base: Qwen2.5-1.5B-Instruct（medical）
        Qwen2.5-1.5B      （legal，在 Base 基座上训练）
         │
  peft.PeftModel.from_pretrained(base, lora_path)
         │
         ▼
  tokenizer.apply_chat_template → model.generate → 返回文本
```

三种 LoRA 类型：

| lora_type | 场景 | 基座 | LoRA 路径 |
|-----------|------|------|-----------|
| `medical` | 医疗健康问诊 | Qwen2.5-1.5B-Instruct | `lora/medical`（约定目录） |
| `legal`   | 法律咨询 | Qwen2.5-1.5B（Base） | `lora/20260309_...`（日期目录） |

**LoRA 评估对比（eval_compare_lora.py）**

用训练集前 N 条做「标准答案 vs LoRA 生成」对比，结果写入 `finetuning/` 目录下的 Markdown 文件：

```
用法:
  python -m service.ai.finetuning.eval_compare_lora --type medical --top 10
  python -m service.ai.finetuning.eval_compare_lora --type legal --top 10

参数:
  --type      medical | legal
  --top       取训练集前几条（默认 10）
  --out       输出文件路径（默认 finetuning/eval_compare_{type}_{date}.md）
  --lora-dir  LoRA 目录路径（不指定则用 lora/medical 或 lora/legal，或最新同名目录）
```

- `generate()` 最大输出 1000 tokens（法律/医疗专业回答篇幅较长，256 tokens 会截断导致评估失真）。

**附加模块**

| 模块 | 内容 |
|------|------|
| `paths.py` | lora 目录约定：`<project>/lora/{日期}_{模型名}/`，提供 `get_run_parent_dir`、`get_latest_lora_dir` 等工具函数 |
| `dataset_legal.py` | 法律领域数据集构建工具 |

**与行业标准方案对比**

| 维度 | Unsloth + LoRA（5090 版） | HF 原生 LoRA（兼容版） | LLaMA-Factory |
|------|--------------------------|----------------------|--------------|
| 速度 | 2-5x 于标准 HF | 基准 | ~2x |
| 显存效率 | 最优（全精度跑 1.5B 只需 ~8GB） | 标准（4bit 时约 4GB） | 中等 |
| 设备兼容性 | CUDA 专用（Flash Attn） | CUDA / MPS / CPU | CUDA 为主 |
| GRPO 支持 | 可配合 TRL GRPOTrainer | 需 TRL | 有限 |
| UI 界面 | 无（纯脚本） | 无 | 有（Web UI） |
| **选型建议** | RTX 5090 / 云 GPU，速度优先 | 本地 Mac / 低显存 CUDA | 需要 UI 配置 |

---

## 第三部分：代码实现深度解析

**决策 1：双版本并行（5090 版 vs HF 兼容版）**

同一领域维护两个训练脚本，而不是单一脚本加参数开关：
- 原因：Unsloth 对 Flash Attention 和 CUDA 有强依赖，MPS/CPU 完全不可用；代码结构也因此差异较大（Unsloth 的 `FastLanguageModel` vs HF 的 `AutoModelForCausalLM + LoraConfig`）。独立文件比 `if unsloth` 分支更清晰。
- 代价：同一逻辑需要维护两份，数据加载代码有重复。

**决策 2：Unsloth 两个 startup patch（5090 版）**

```python
# Patch 1：修复 transformers tokenizer bug
# transformers 某版本 _config.model_type 对 dict 报 AttributeError
old = "if _is_local and _config.model_type not in ["
new = 'if _is_local and getattr(_config, "model_type", None) not in ['

# Patch 2：跳过 SFTTrainer eos_token 词表校验
# Qwen2+unsloth 的 eos_token=<|im_end|> 不在标准词表，导致 SFTTrainer 报错
# 用括号计数精确定位 raise ValueError 块，替换为 pass
```

这两个 patch 在脚本启动时直接修改已安装的 transformers / trl 源码文件。
原因：Unsloth + Qwen2 + 新版 trl 之间存在版本兼容问题，patch 是最小侵入方案，不需要降级依赖。
代价：直接修改 site-packages 文件，跨版本升级时可能失效，需重新验证。

**决策 3：LoRA alpha 设为 2× rank（5090 版）**

```python
r=16, lora_alpha=32   # medical_5090 和 legal_5090
r=16, lora_alpha=16   # medical.py（默认 1:1）
```

`lora_alpha / r` 决定 LoRA 的有效学习率放大倍数。5090 版设为 2:1（强化学习信号），HF 兼容版设为 1:1（更保守，适合小 batch 的本地训练）。

**决策 4：法律 LoRA 使用 Base 基座（而非 Instruct）**

`finetuning.py` 中 legal 类型使用 `Qwen2.5-1.5B`（Base）而非 Instruct 版本：
```python
LEGAL_BASE_MODEL_NAME = "Qwen2.5-1.5B"  # legal 用 Base
# medical 用 Qwen2.5-1.5B-Instruct
```

原因：法律 LoRA 训练时使用 Base 基座（`legal.py`），推理时必须与训练时基座一致，否则 Adapter 权重偏移会导致输出混乱。
代价：Base 模型没有对话格式的预训练，纯法律微调需要数据质量更高。

**决策 5：推理服务懒加载 + 线程锁**

```python
_model_tokenizer = {}  # lora_type -> (model, tokenizer)，已加载的模型缓存

def get_model(lora_type="medical"):
    with _model_tokenizer_lock:           # 防止并发时重复加载
        if lora_type in _model_tokenizer:
            return _model_tokenizer[lora_type]
        # 首次调用时加载 base + LoRA，约 3-5 秒
        ...
        _model_tokenizer[lora_type] = (model, tokenizer)
```

原因：模型加载耗时 3-5 秒且占用约 3-6GB 内存，不适合在 Flask 启动时就全部加载；懒加载保证只有首次请求才支付加载代价，后续请求直接复用。
代价：该 lora_type 的第一次请求延迟高；多 lora_type 同时首次调用时互相阻塞（锁粒度为全局）。

**决策 6：eval_compare_lora 中 max_new_tokens=1000**

法律/医疗专业回答篇幅较长，标准答案平均 300-800 汉字。256 tokens（约 160 汉字）会造成严重截断，使 LoRA 生成答案看起来比标准答案差，但实际上是被强制截止。1000 tokens 可还原完整输出能力，评估结果更接近真实。
代价：单条推理时间从约 3-5 秒增加至 10-20 秒，批量评估耗时增加。

---

## 第四部分：应用场景与实战

**使用场景**

- **医疗健康问诊**：用户通过 `POST /ai/finetuning/chat` 发送医疗问题，服务调用 Qwen2.5-1.5B + 医疗 LoRA 回答（研究/演示用途，非正式医疗建议）
- **法律咨询**：同接口，切换 `lora_type=legal`，回答中国法律相关问题，可引用法律条文
- **微调效果验证**：训练完成后运行 `eval_compare_lora.py`，对比标准答案与 LoRA 生成，人工评估微调效果

**环境依赖**

```bash
# 5090 版（Unsloth 路线）
pip install unsloth torch transformers trl
# unsloth 要求 CUDA 11.8+，Flash Attention 2 自动安装

# HF 兼容版（本地 Mac / 低显存 CUDA）
pip install torch transformers peft trl bitsandbytes datasets pandas
# Mac MPS 不需要 bitsandbytes（4bit 不可用）

# 评估脚本
pip install torch transformers peft
# 需要已训练好的 LoRA 权重放到 lora/ 目录
```

**代码示例：通过 HTTP 接口调用 LoRA 模型**

```python
import requests

# 医疗问诊
res = requests.post("http://localhost:5000/ai/finetuning/chat", json={
    "messages": [{"role": "user", "content": "我最近总是感觉头晕，应该怎么办？"}],
    "lora_type": "medical",
})
print(res.json()["data"]["content"])

# 法律咨询
res = requests.post("http://localhost:5000/ai/finetuning/chat", json={
    "messages": [{"role": "user", "content": "公司突然裁员，没有提前通知也没有赔偿，合法吗？"}],
    "lora_type": "legal",
})
print(res.json()["data"]["content"])

# 查询可用的 LoRA 选项
opts = requests.get("http://localhost:5000/ai/finetuning/lora-options")
print(opts.json()["data"])  # ["medical", "legal"]
```

**代码示例：直接运行 5090 训练脚本**

```bash
# RTX 5090 上训练医疗 LoRA（全量 HuatuoGPT 数据）
cd <project_root>
python -m service.ai.finetuning.medical_5090
# 完成后权重保存到 lora/日期_Qwen2.5-1.5B-Instruct-medical-huatuo-5090/

# 训练法律 LoRA（全量 DISC-Law-SFT 数据）
python -m service.ai.finetuning.legal_5090
```

**代码示例：本地 Mac 兼容版训练（小数据集验证）**

```python
# medical.py 中调整参数进行本地调试：
MAX_TRAIN_SAMPLES = 100   # 取 100 条做快速验证
# 代码自动检测 MPS/CPU，bf16→fp16，关闭 gradient_checkpointing
```

**常见问题**

- **`CUDA out of memory`（5090 版）**：5090 版 `load_in_4bit=False` 全精度运行，1.5B 约占 8-10GB，batch=32/48 时峰值 ~25GB。若 OOM 可降 `per_device_train_batch_size` 或在 `legal_5090.py` 中加 `USE_4BIT=1`。
- **`AttributeError: 'dict' has no attribute 'model_type'`**：transformers tokenizer bug，5090 版脚本的 `_fix_tokenizer_dict_bug()` 会在启动时自动修复；`medical.py` / `finetuning.py` 用 `PreTrainedTokenizerFast(tokenizer_file=...)` 绕过。
- **`eos_token not in vocabulary`（SFTTrainer）**：Qwen2 + Unsloth 兼容问题，`_fix_sft_trainer_eos_check()` 会自动 patch trl 源码，或手动 `add_special_tokens({"eos_token": "<|im_end|>"})` 并 `tokenizer.eos_token_id = ...`。
- **推理服务报 `LoRA 适配器目录不存在`**：检查 `LORA_TYPE_DIRS` 中的目录名是否与实际 `lora/` 下的目录一致；`legal` 类型的目录名含日期，须手动更新 `finetuning.py` 中的 `LORA_TYPE_DIRS["legal"]`。
- **MPS 训练 NaN 或梯度异常**：关闭 `gradient_checkpointing`（MPS 与 meta device 冲突），`medical.py` 已处理（`gradient_checkpointing=(device == "cuda")`）。

---

## 第五部分：优缺点评估与未来展望

**优势**

- Unsloth 5090 版在 32GB 显存上可全精度 bf16 训练，无量化损失，收敛速度快
- 双版本设计兼顾 GPU 服务器（高吞吐训练）和本地 Mac/低显存机器（兼容调试）
- 推理服务（`finetuning.py`）无缝集成主服务，懒加载设计不增加启动时间
- 覆盖医疗、法律两个领域，架构可复用，新增领域只需添加数据集 + 训练脚本 + 注册 `LORA_TYPE_DIRS`

**已知局限**

- 两个领域各有两份训练脚本，数据加载逻辑有重复，维护成本随领域增加线性增长
- `finetuning.py` 中 `LORA_TYPE_DIRS["legal"]` 含日期硬编码（`"20260309_..."`），新训练完成后需手动更新
- 训练完成到推理服务可用需手动操作（训练 → 复制到约定目录 → 重启服务）
- `finetuning.py` 的全局锁（`_model_tokenizer_lock`）在多 lora_type 并发首次请求时会串行化

**演进建议**

- 短期：将 `LORA_TYPE_DIRS["legal"]` 改为使用 `get_latest_lora_dir()` 动态解析，消除硬编码日期
- 中期：抽取通用 LoRA 训练框架（数据加载 + 格式化 + 训练参数 + 保存），各领域只需提供数据加载函数和 system prompt，消除重复代码
- 长期：构建自动化微调流水线：数据上传 → 触发训练 → 训练完成后自动注册到推理服务 → A/B 对比基座模型

**行业前沿**

- **DoRA（权重分解 LoRA）**：将权重分解为幅度和方向两部分分别训练，比标准 LoRA 效果更好，Unsloth 已支持
- **ORPO（偏好优化 LoRA）**：SFT 和偏好对齐（RLHF）一步完成，比 GRPO 更简单，适合专业领域问答对齐
- **持续预训练（Continual Pretraining）**：在领域语料（医学文献、法律条文）上做预训练再 SFT，比纯 SFT 的领域理解更深，是专业模型的终极方案

---

## 变更记录

| 日期 | 变更说明 |
|------|----------|
| 2026-03-11 | 全文重写：旧脚本（Qwen2.5-7B 系列、视觉微调 VL、GRPO/R1）已全部移除；记录当前实际架构（medical_5090/legal_5090/medical/legal 四个训练脚本 + finetuning.py 推理服务 + eval_compare_lora.py + paths.py）。 |
| 2026-03-11 | 删除空气小猪（airpig）LoRA 全部逻辑：移除 `finetuning.py` 中的 `AIRPIG_SYSTEM_PROMPT`、`_inject_airpig_system()`、`LORA_TYPE_DIRS["airpig"]`、`LORA_OPTIONS` 中的 airpig 条目；更新文档相关描述。 |
