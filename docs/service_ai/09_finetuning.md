# 模型微调实验层（Finetuning）

> 文件：`service/ai/finetuning/`（Qwen2_5_(7B)_alpaca.py、Qwen2_5_(7B)_medical.py、Qwen2_5_(7B)_R1.py、qwen_vl_car_insurance_train.py、image_svd/、MovieLens/ 等）  
> 生成日期：2026-02-26

---

## 第一部分：技术背景与演进

**问题背景**

通用大模型（GPT-4、Qwen-7B）在通用任务上表现优秀，但面对高度专业化场景（医疗诊断、法律条文、特定格式输出）时，它们的回答往往不够精确、不符合行业术语规范，或者需要大量 Prompt 工程才能勉强达标。微调（Finetuning）通过在领域数据上继续训练模型，让模型"内化"领域知识和输出风格，从根本上解决通用模型的领域适应问题。

**核心概念**

- **SFT（监督式微调）**：用"问题-答案"对训练模型学会特定格式和领域知识，是最常见的微调方式。
- **LoRA（低秩适配器）**：不改动原始权重，只在关键层旁边插入小矩阵（低秩矩阵）进行训练。7B 参数模型用 LoRA 微调只需更新约 0.1% 的参数，显存需求从 80GB 降至 10GB 以内。
- **GRPO（生成奖励配对优化）**：DeepSeek R1 提出的方法，通过多目标奖励函数（答案正确性 + 格式合规性 + 推理链质量）引导模型学会"先推理后作答"的思考风格。
- **Unsloth**：开源微调加速框架，通过 Flash Attention 2 + Triton 内核优化，使 SFT 速度提升 2-5 倍，显存占用减少约 60%。

**演进脉络**

| 阶段 | 方案 | 特点 |
|------|------|------|
| 早期 | 全量参数微调（Full Fine-tuning） | 效果好，但需要数十 GB 显存，成本高 |
| 2021 | Adapter 方法 | 插入小模块，参数量少，但速度有损 |
| 2021 | **LoRA**（微软） | 低秩矩阵，参数量极少，速度接近全量 |
| 2023 | QLoRA | LoRA + 4bit 量化，消费级 GPU 可微调 7B+ 模型 |
| 2024 | **GRPO / DPO** | 奖励信号驱动，训练推理能力而非仅输出格式 |
| 2024 | **Unsloth** | 工程加速，让 LoRA 训练快 2-5 倍 |

**本模块的定位**

`finetuning/` 目录是独立于主服务的实验脚本集合，不提供 HTTP 接口，直接在 GPU 机器上运行训练。涵盖三种典型微调场景和一套线性代数/推荐系统实验，是理解"如何从通用模型定制领域模型"的完整参考实现。

---

## 第二部分：架构剖析

**四个微调脚本的技术对比**

| 脚本 | 模型 | 任务类型 | 训练器 | 数据来源 | 输出 |
|------|------|---------|-------|---------|------|
| `qwen_vl_car_insurance_train.py` | Qwen2.5-VL-3B | 视觉+文本 SFT | SFTTrainer + VisionDataCollator | qwen-vl-train.xlsx | car_insurance_lora_model |
| `Qwen2_5_(7B)_medical.py` | Qwen2.5-7B | 文本 SFT | SFTTrainer | Data_数据/\*.csv | lora_model_medical |
| `Qwen2_5_(7B)_alpaca.py` | Qwen2.5-7B | 文本 SFT | SFTTrainer | alpaca-cleaned | lora_model |
| `Qwen2_5_(7B)_R1.py` | Qwen2.5-7B | GRPO 推理优化 | GRPOTrainer | GSM8K | grpo_saved_lora |

**SFT 微调完整流程**

```
1. 加载基座模型（4bit 量化，减少显存）
   FastLanguageModel.from_pretrained("Qwen2.5-7B-Instruct", load_in_4bit=True)

2. 注入 LoRA Adapter
   get_peft_model(model, r=16, lora_alpha=16,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", ...])

3. 准备数据集（格式：{text: "Prompt + EOS_TOKEN"}）
   formatting_prompts_func → 拼 system + user + assistant

4. 训练（SFTTrainer）
   num_train_epochs=3 / max_steps=60
   per_device_train_batch_size=2
   gradient_accumulation_steps=4

5. 保存 LoRA 权重（约 100-300MB）
   model.save_pretrained("lora_model_medical")

6. 推理测试
   FastLanguageModel.for_inference(model)  # 开启推理优化
   model.generate(...)
```

**GRPO 推理微调流程（R1.py）**

```
训练目标：让 Qwen2.5-7B 输出 <reasoning>...</reasoning><answer>...</answer> 格式

奖励函数组合（多目标）：
  ┌── correctness_reward_func   答案正确 → +2.0
  ├── int_reward_func           答案为整数 → +0.5
  ├── strict_format_reward_func 严格 XML 格式 → +0.5
  ├── soft_format_reward_func   宽松格式匹配 → +0.5
  └── xmlcount_reward_func      XML 标签完整性细粒度得分

GRPOTrainer：
  num_generations=6    每个 prompt 生成 6 个样本对比
  max_steps=250
  使用 vLLM fast_generate 加速 rollout
```

**视觉微调（qwen_vl_car_insurance_train.py）**

```
输入数据格式（Excel → 对话格式）：
  user:      [TextPart: "请从图中提取关键信息"] + [ImagePart: 车辆里程表图片]
  assistant: "里程数：12,345 公里，仪表盘状态正常..."

关键配置：
  finetune_vision_layers=True    同时微调视觉编码器
  finetune_language_layers=True  同时微调语言解码器
  UnslothVisionDataCollator      处理混合文本+图片的 Batch 整理
```

**附加实验模块**

| 模块 | 内容 |
|------|------|
| `image_svd/image_svd.py` | 用 NumPy SVD（奇异值分解）实现图片压缩，演示 LoRA 的数学基础 |
| `MovieLens/ALS.py` | 交替最小二乘（ALS）矩阵分解推荐系统，同样是低秩矩阵应用 |
| `dataset_legal.py` | 法律领域数据集构建工具 |

`image_svd` 和 `MovieLens/ALS` 是 LoRA 的数学概念演示——LoRA 本质上是对权重矩阵做低秩近似，和 SVD 图片压缩、ALS 矩阵分解同属"用低秩矩阵近似高维矩阵"的思想。

**与行业标准方案对比**

| 维度 | Unsloth + LoRA（本项目） | HuggingFace PEFT + Trainer | LLaMA-Factory |
|------|------------------------|--------------------------|--------------|
| 速度 | 2-5x 于标准 HF | 基准 | ~2x（依赖 HF Trainer） |
| 显存效率 | 最优（~60% 节省） | 标准 | 中等 |
| 支持模型 | Qwen/Llama/Mistral 等热门 | 几乎所有 HF 模型 | Qwen/Llama 等 |
| GRPO 支持 | 原生（配合 TRL GRPOTrainer） | 需 TRL | 有限 |
| UI 界面 | 无（纯脚本） | 无 | 有（Web UI） |
| **选型建议** | 速度优先、Qwen 系列、资源受限 | 通用兼容、模型多样性 | 需要 UI 配置、不想写代码 |

---

## 第三部分：代码实现深度解析

**核心设计决策**

**决策 1：4bit 量化 + LoRA 的组合（QLoRA）**  
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen2.5-7B-Instruct",
    load_in_4bit=True,         # 4bit NF4 量化，7B 显存约 5-6GB
    dtype=None,                # 自动选择（bf16/fp16）
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                      # LoRA 秩：值越大参数越多，默认 16 是均衡选择
    lora_alpha=16,             # 通常设为等于 r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
```
4bit 量化让 7B 模型从约 14GB 显存需求降至 5-6GB，单块 T4（16GB 显存）即可训练。LoRA 只训练约 0.1% 的参数，保留原始模型能力的同时学习领域知识。

**决策 2：医疗数据集的多编码容错**  
```python
for enc in ["utf-8", "gbk", "gb18030", "utf-8-sig"]:
    try:
        df = pd.read_csv(f, encoding=enc)
        break
    except UnicodeDecodeError:
        continue
```
医疗数据来自多个来源，编码不统一。逐一尝试常见编码而不是要求统一格式，提升了数据加载的鲁棒性。同时对过长 QA 对（>200 字）做过滤，避免训练数据噪声。

**决策 3：GRPO 多目标奖励分层设计**  
```python
reward_funcs = [
    correctness_reward_func,   # 最高权重：结果正确最重要
    int_reward_func,           # 辅助：答案格式（整数）
    strict_format_reward_func, # 格式合规性（严格 XML）
    soft_format_reward_func,   # 格式合规性（宽松匹配）
    xmlcount_reward_func,      # 细粒度标签完整性
]
```
多目标奖励确保模型同时优化"结果正确"和"格式合规"，防止模型为了得高分只学格式忽略正确性（或反之）。

**决策 4：`image_svd` 演示 LoRA 的数学直觉**  
```python
# SVD 分解图片矩阵，只保留前 k 个奇异值
U, S, Vt = np.linalg.svd(img_channel)
reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
# k 越小，压缩率越高，图片越模糊
# LoRA 的核心思想与此完全相同：用低秩矩阵 A×B 近似全秩权重矩阵 W
```
这个可视化实验让"为什么 LoRA 只需要小矩阵就能有效调整大模型行为"变得直观可理解。

---

## 第四部分：应用场景与实战

**使用场景**

- **医疗领域专家模型**：Qwen2.5-7B 在医疗 QA 数据上 SFT，输出专业医疗建议（研究/演示用途，非正式医疗建议）
- **视觉理解应用**：Qwen2.5-VL-3B 在车险图片上微调，自动从汽车照片提取关键信息，支持保险核保自动化
- **推理能力增强**：用 GRPO 让 Qwen2.5-7B 学会 DeepSeek-R1 风格的思维链推理，提升数学/逻辑任务准确率
- **概念学习**：`image_svd` 和 `MovieLens/ALS` 是理解 LoRA 低秩近似思想的最佳可视化入门

**环境依赖**

```bash
# 主要依赖（建议 AutoDL 等 GPU 云服务器）
pip install unsloth torch transformers trl peft datasets
pip install pandas pillow openpyxl  # VL 微调额外依赖

# 模型路径（脚本内为 AutoDL 云服务器路径，本地需修改）
# /root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct
# /root/autodl-tmp/datasets/gsm8k
```

**代码示例**

```python
# 加载 LoRA 微调后的医疗模型推理
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "lora_model_medical",    # 微调保存的路径
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # 开启 2x 推理加速

inputs = tokenizer(
    [medical_prompt.format("请问高血压的饮食注意事项有哪些？", "")],
    return_tensors="pt"
).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

**常见问题**

- **`CUDA out of memory`**：减小 `per_device_train_batch_size` 至 1，增大 `gradient_accumulation_steps` 到 8；或将 `r` 从 16 降至 8。
- **路径找不到**：脚本中路径为 AutoDL 云服务器路径（`/root/autodl-tmp/...`），本地运行需全部改为本地路径。
- **GRPO 训练不稳定**：`num_generations=6` 在显存不足时可降为 4；`max_steps=250` 可先用 50 步验证奖励曲线是否上升。

---

## 第五部分：优缺点评估与未来展望

**优势**

- Unsloth 框架加速显著，同等显存下可训练更大批量或更多步
- QLoRA（4bit + LoRA）让消费级 GPU 可训练 7B 模型，极大降低微调门槛
- 涵盖三种微调范式（SFT/多模态 SFT/GRPO），覆盖主流场景
- `image_svd` + `ALS` 提供了理解 LoRA 数学直觉的可视化辅助

**已知局限**

- 脚本路径硬编码为 AutoDL 服务器路径，不适合直接在本地运行
- 训练脚本与主服务完全解耦——微调后的模型无法直接注入到 `chat.py` 或 `rag.py` 的推理流程
- 缺少训练指标可视化（TensorBoard/WandB 集成）
- 医疗/法律场景的微调模型尚未集成到主服务中

**演进建议**

- 短期：将路径改为环境变量配置，支持本地 + AutoDL 两套环境；添加 WandB 训练指标追踪
- 中期：将微调后的 LoRA 模型通过 Ollama 格式（GGUF）导出，无缝集成到 `ollama_chat.py` 的 `DEFAULT_MODEL` 中
- 长期：构建微调流水线：数据上传 → 自动数据清洗 → 触发训练（AutoDL API） → 模型自动部署 → A/B 测试对比基座模型

**行业前沿**

- **DoRA（权重分解 LoRA）**：将权重分解为幅度和方向两部分分别训练，比标准 LoRA 效果更好，Unsloth 已支持
- **ORPO（偏好优化 LoRA）**：SFT 和偏好对齐（RLHF）一步完成，比 GRPO 更简单
- **LoRA+ 和 Flora**：改进 LoRA 的优化器学习率策略，在相同参数量下收敛更快、效果更好
- **持续预训练（Continual Pretraining）**：在领域语料（医学文献、法律条文）上做预训练再 SFT，比纯 SFT 的领域理解更深，是专业模型的终极方案
