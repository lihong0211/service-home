#!/usr/bin/env python
# coding: utf-8

# ### 使用 Unsloth 框架对 Qwen2.5-7B 模型进行微调的示例代码
# ### 本代码可以在免费的 Tesla T4 Google Colab 实例上运行 https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb

# In[1]:


# 导入必要的库
import os
from unsloth import FastLanguageModel
import torch

# 路径配置：本地模型 models/Qwen/Qwen2.5-7B-Instruct；Jupyter 无 __file__ 时用当前工作目录
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))
BASE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "Qwen", "Qwen2.5-7B-Instruct")
if not os.path.isdir(BASE_MODEL_PATH):
    BASE_MODEL_PATH = os.path.join(_SCRIPT_DIR, "models", "Qwen", "Qwen2.5-7B-Instruct")  # 备选：脚本同目录下
MEDICAL_DATA_DIR = os.path.join(_SCRIPT_DIR, "【数据集】中文医疗数据")
LORA_SAVE_DIR = os.path.join(_SCRIPT_DIR, "lora_model_medical")  # 微调后 LoRA 保存路径

# 设置模型参数
max_seq_length = 2048  # 设置最大序列长度，支持 RoPE 缩放
dtype = None  # 数据类型，None 表示自动检测。Tesla T4 使用 Float16，Ampere+ 使用 Bfloat16
load_in_4bit = True  # 使用 4bit 量化来减少内存使用

# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


# In[2]:


# 添加LoRA适配器，只需要更新1-10%的参数
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA秩，建议使用8、16、32、64、128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],  # 需要应用LoRA的模块
    lora_alpha = 16,  # LoRA缩放因子
    lora_dropout = 0,  # LoRA dropout率，0为优化设置
    bias = "none",    # 偏置项设置，none为优化设置
    use_gradient_checkpointing = "unsloth",  # 使用unsloth的梯度检查点，可减少30%显存使用
    random_state = 3407,  # 随机种子
    use_rslora = False,  # 是否使用rank stabilized LoRA
    loftq_config = None,  # LoftQ配置
)


# ### 数据准备

# In[3]:


import pandas as pd
from datasets import Dataset

# 定义医疗对话的提示模板
medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""

# 获取结束标记
EOS_TOKEN = tokenizer.eos_token

def read_csv_with_encoding(file_path):
    """尝试使用不同的编码读取CSV文件"""
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法使用任何编码读取文件: {file_path}")

def load_medical_data(data_dir):
    """加载医疗对话数据"""
    data = []
    departments = {
        'IM_内科': '内科',
        'Surgical_外科': '外科',
        'Pediatric_儿科': '儿科',
        'Oncology_肿瘤科': '肿瘤科',
        'OAGD_妇产科': '妇产科',
        'Andriatria_男科': '男科'
    }
    
    # 遍历所有科室目录
    for dept_dir, dept_name in departments.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            print(f"目录不存在: {dept_path}")
            continue
            
        print(f"\n处理{dept_name}数据...")
        
        # 获取该科室下的所有CSV文件
        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_path = os.path.join(dept_path, csv_file)
            print(f"正在处理文件: {csv_file}")
            
            try:
                # 读取CSV文件
                df = read_csv_with_encoding(file_path)
                
                # 打印列名，帮助调试
                print(f"文件 {csv_file} 的列名: {df.columns.tolist()}")
                
                # 处理每一行数据
                for _, row in df.iterrows():
                    try:
                        # 获取问题和回答（尝试不同的列名）
                        question = None
                        answer = None
                        
                        # 尝试不同的列名
                        if 'question' in row:
                            question = str(row['question']).strip()
                        elif '问题' in row:
                            question = str(row['问题']).strip()
                        elif 'ask' in row:
                            question = str(row['ask']).strip()
                            
                        if 'answer' in row:
                            answer = str(row['answer']).strip()
                        elif '回答' in row:
                            answer = str(row['回答']).strip()
                        elif 'response' in row:
                            answer = str(row['response']).strip()
                        
                        # 过滤无效数据
                        if not question or not answer:
                            continue
                            
                        # 限制长度
                        if len(question) > 200 or len(answer) > 200:
                            continue
                            
                        # 添加到数据列表
                        data.append({
                            "instruction": "请回答以下医疗相关问题",
                            "input": question,
                            "output": answer
                        })
                        
                    except Exception as e:
                        print(f"处理数据行时出错: {e}")
                        continue
                        
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue
    
    # 验证数据
    if not data:
        raise ValueError("没有成功处理任何数据！")
        
    print(f"\n成功处理 {len(data)} 条数据")
    return Dataset.from_list(data)

def formatting_prompts_func(examples):
    """格式化提示"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = medical_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# 加载医疗数据集
dataset = load_medical_data(MEDICAL_DATA_DIR)
dataset = dataset.map(formatting_prompts_func, batched=True)


# ### 模型训练

# In[4]:


# 设置训练参数和训练器
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 定义训练参数
training_args = TrainingArguments(
        per_device_train_batch_size = 2,  # 每个设备的训练批次大小
        gradient_accumulation_steps = 4,  # 梯度累积步数
        warmup_steps = 5,  # 预热步数
        #max_steps = 60,  # 最大训练步数
        max_steps = -1,  # 不使用max_steps
        num_train_epochs = 3,  # 训练3个epoch
        learning_rate = 2e-4,  # 学习率
        fp16 = not is_bfloat16_supported(),  # 是否使用FP16
        bf16 = is_bfloat16_supported(),  # 是否使用BF16
        logging_steps = 1,  # 日志记录步数
        optim = "adamw_8bit",  # 优化器
        weight_decay = 0.01,  # 权重衰减
        lr_scheduler_type = "linear",  # 学习率调度器类型
        seed = 3407,  # 随机种子
        output_dir=os.path.join(_SCRIPT_DIR, "outputs"),  # 输出目录（脚本同目录下）
        report_to = "none",  # 报告方式
    )

# 创建SFTTrainer实例
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,  # 对于短序列可以设置为True，训练速度提升5倍
    args = training_args,
)


# In[5]:


# 显示当前GPU内存状态
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[6]:


# 开始训练
trainer_stats = trainer.train()


# In[22]:


# 显示训练后的内存和时间统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# ### 模型推理

# In[15]:


# 模型推理示例
def generate_medical_response(question):
    """生成医疗回答"""
    FastLanguageModel.for_inference(model)  # 启用原生2倍速推理
    inputs = tokenizer(
        [medical_prompt.format(question, "")],
        return_tensors="pt"
    ).to("cuda")
    
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    
# 测试问题
test_questions = [
    "我最近总是感觉头晕，应该怎么办？",
    "感冒发烧应该吃什么药？",
    "高血压患者需要注意什么？"
]

for question in test_questions:
    print("\n" + "="*50)
    print(f"问题：{question}")
    print("回答：")
    generate_medical_response(question) 


# ### 微调模型保存
# **[注意]** 这里只是LoRA参数，不是完整模型。

# In[23]:


# 保存模型（保存到脚本同目录下 lora_model_medical）
model.save_pretrained(LORA_SAVE_DIR)
tokenizer.save_pretrained(LORA_SAVE_DIR)


# In[24]:


# 加载保存的模型进行推理
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model",  # 训练时使用的模型
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # 启用原生2倍速推理
    
question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question) 


# In[25]:


# 加载保存的模型进行推理
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_SAVE_DIR,  # 保存的模型
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # 启用原生2倍速推理
    
question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question) 


# In[26]:


# 加载保存的模型进行推理
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_PATH,
        adapter_name=LORA_SAVE_DIR,  # LoRA权重
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # 启用原生2倍速推理

question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question)


# In[27]:


question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question)


# ### 效果对比检验：Base vs 微调后 LoRA
# 用同一批测试问题分别跑「仅基座」和「基座+LoRA」，并排对比输出，便于检验微调效果。

# In[28]:


def generate_response_no_stream(model, tokenizer, question, max_new_tokens=256):
    """不流式生成，直接返回完整回答文本（用于对比）"""
    prompt = medical_prompt.format(question, "")
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # 只返回新生成部分
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def compare_base_vs_finetuned(
    base_model_path,
    lora_adapter_path,
    test_questions,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    output_file=None,
):
    """
    对比「仅基座」与「基座+LoRA」在同一批问题上的回答，并排打印并可选保存到文件。
    """
    from unsloth import FastLanguageModel

    results = []

    # 1) 仅基座
    print("加载基座模型（无 LoRA）...")
    model_base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model_base)
    for q in test_questions:
        ans = generate_response_no_stream(model_base, tokenizer, q)
        results.append({"question": q, "base": ans, "lora": None})
    del model_base
    torch.cuda.empty_cache()

    # 2) 基座 + LoRA
    print("加载基座 + 医疗 LoRA...")
    model_lora, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        adapter_name=lora_adapter_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model_lora)
    for i, q in enumerate(test_questions):
        results[i]["lora"] = generate_response_no_stream(model_lora, tokenizer, q)
    del model_lora
    torch.cuda.empty_cache()

    # 3) 并排打印
    print("\n" + "=" * 80)
    print("【效果对比】Base vs 微调后 LoRA")
    print("=" * 80)
    for r in results:
        print("\n--- 问题 ---")
        print(r["question"])
        print("\n--- 基座回答 ---")
        print(r["base"])
        print("\n--- 微调后回答 ---")
        print(r["lora"])
        print("-" * 40)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write("问题: " + r["question"] + "\n")
                f.write("基座: " + r["base"] + "\n")
                f.write("微调: " + r["lora"] + "\n\n")
        print(f"\n结果已保存到: {output_file}")

    return results


# 使用示例：对比基座与医疗 LoRA
LORA_MEDICAL = LORA_SAVE_DIR
compare_questions = [
    "我最近总是感觉头晕，应该怎么办？",
    "感冒发烧应该吃什么药？",
    "高血压患者需要注意什么？",
]
# 取消下面注释以运行对比（会先后加载基座和基座+LoRA，显存占用约一倍）
# compare_base_vs_finetuned(
#     BASE_MODEL_PATH,
#     LORA_MEDICAL,
#     compare_questions,
#     max_seq_length=max_seq_length,
#     dtype=dtype,
#     load_in_4bit=load_in_4bit,
#     output_file="compare_base_vs_medical_lora.txt",
# )


# ### 热插拔 Checkpoint：不重载基座，切换不同 LoRA
# 方式一：多 adapter 并存，用 set_adapter 切换；
# 方式二：用 PEFT 的 hotswap_adapter 原地替换当前 adapter 权重（同结构 LoRA 可快速切换）。

# In[29]:


def load_base_then_adapters(base_model_path, adapter_paths_dict, max_seq_length=2048, dtype=None, load_in_4bit=True):
    """
    只加载一次基座，挂载多个 LoRA adapter，返回 (model, tokenizer)。
    adapter_paths_dict: {"adapter_name": "path_or_name"}，例如 {"medical": "lora_model_medical", "alpaca": "lora_model"}
    第一个 adapter 随基座一起加载，其余用 load_adapter 挂上。
    """
    from unsloth import FastLanguageModel

    items = list(adapter_paths_dict.items())
    if not items:
        raise ValueError("adapter_paths_dict 至少需要一个 adapter")
    first_name, first_path = items[0]
    # 基座 + 第一个 adapter（Unsloth 要求至少带一个 adapter 才会变成 PEFT 模型）
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        adapter_name=first_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    peft_model = model.model if hasattr(model, "model") else model
    # 第一个已加载，通常名为 "default"，可改名为 first_name 或保持 default；其余挂载
    for name, path in items[1:]:
        peft_model.load_adapter(path, adapter_name=name)
    # 若希望第一个也按名字切换，可再 load_adapter(first_path, adapter_name=first_name)，会重复一份；一般用 default 即可
    return model, tokenizer


def switch_adapter(model, adapter_name):
    """在已挂载的多个 adapter 间切换（热插拔：不重载基座）。"""
    peft_model = model.model if hasattr(model, "model") else model
    peft_model.set_adapter(adapter_name)


# 热插拔示例：同一基座下切换 医疗 LoRA / 其他 LoRA
# 第一个 adapter 在 PEFT 里名为 "default"，其余用你传入的 key 作为名字
if False:  # 改为 True 并确保有对应 checkpoint 路径时运行
    model, tokenizer = load_base_then_adapters(
        BASE_MODEL_PATH,
        {"default": LORA_SAVE_DIR, "alpaca": "lora_model"},  # 第一个随基座加载，名为 default
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    switch_adapter(model, "default")   # 使用医疗 LoRA
    generate_medical_response("我最近总是感觉头晕，应该怎么办？")
    switch_adapter(model, "alpaca")    # 切换到另一 LoRA
    generate_medical_response("写一首短诗")


# In[30]:


def hotswap_lora_inplace(model, new_adapter_path, adapter_name="default", device="cuda"):
    """
    不重载基座，把当前已加载的 LoRA 权重原地替换为另一份 LoRA（同结构）。
    适合同一基座、多个 checkpoint 轮流对比（如 step-100 vs step-500）。
    """
    from peft.utils.hotswap import hotswap_adapter

    peft_model = model.model if hasattr(model, "model") else model
    hotswap_adapter(peft_model, new_adapter_path, adapter_name=adapter_name, torch_device=device)


# 使用示例：已加载 base + lora_model_medical 后，想换成同结构的另一 checkpoint
# hotswap_lora_inplace(model, "outputs/checkpoint-500", adapter_name="default", device="cuda")
