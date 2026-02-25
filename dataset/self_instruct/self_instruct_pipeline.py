# dataset/self_instruct/self_instruct_pipeline.py
"""
Self-Instruct 数据生成流水线：从少量种子指令出发，用大模型批量生成新指令。
支持 OpenAI 兼容 API（含 Qwen / 本地部署），输出 JSONL，可对接 Alpaca 微调格式。
"""
import json
import os
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    from .filters import InstructionFilter, default_filter_config
except ImportError:
    from filters import InstructionFilter, default_filter_config

# 默认使用 openai 库调用（兼容 OpenAI / Qwen / 本地 base_url）
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


SEED_FORMAT_HINT = """
种子文件应为 JSONL，每行一个 JSON，包含 "instruction" 字段，例如：
{"id": "1", "instruction": "将下面的中文翻译成英文。"}
"""

GENERATION_PROMPT = """你是一个指令设计专家。请根据下面给出的一组「任务指令」示例，生成 {num} 条全新的、多样化的任务指令。

要求：
- 每条指令用一句话描述一个用户可能向助手提出的任务，可以是中文或英文。
- 不要与示例重复或高度相似，主题和表述要有变化。
- 每条指令单独一行，不要编号、不要引号包裹。

示例指令：
{examples}

请直接输出 {num} 条新指令，每条一行："""


def load_seed_instructions(seed_path: str) -> List[dict]:
    """从 JSONL 加载种子指令。每行需包含 instruction 字段。"""
    instructions = []
    path = Path(seed_path)
    if not path.exists():
        raise FileNotFoundError(f"种子文件不存在: {seed_path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "instruction" in item:
                    instructions.append(item)
            except json.JSONDecodeError:
                continue
    return instructions


def load_all_instructions(jsonl_path: str) -> List[str]:
    """从输出 JSONL 中加载已有指令列表（用于去重/相似度过滤）。"""
    path = Path(jsonl_path)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "instruction" in item:
                    out.append(item["instruction"])
            except json.JSONDecodeError:
                continue
    return out


def _create_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    """创建 OpenAI 兼容客户端。优先环境变量 QWEN_API_KEY / OPENAI_API_KEY，OPENAI_BASE_URL。"""
    if OpenAI is None:
        raise ImportError("请安装 openai: pip install openai")
    key = api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base = base_url or os.environ.get("OPENAI_BASE_URL")
    if not key and not base:
        raise ValueError("请设置 QWEN_API_KEY 或 OPENAI_API_KEY；若用本地服务请设置 OPENAI_BASE_URL 和对应 API Key")
    return OpenAI(api_key=key or "dummy", base_url=base) if base else OpenAI(api_key=key)


def _call_llm(
    client: Any,
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """调用 LLM 生成文本。"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    return text.strip()


def parse_generated_lines(text: str) -> List[str]:
    """从模型输出中解析出每一条指令（按行拆分，去掉空行和明显非指令行）。"""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # 去掉行首编号如 "1. " "2)"
        for sep in (". ", ")、", "）", ")", "、", ":", "："):
            if line.startswith(sep) or (len(line) > 2 and line[1] == sep[0]):
                for i, c in enumerate(line):
                    if c not in "0123456789.)、）:：、 ":
                        line = line[i:].strip()
                        break
                break
        if len(line) >= 3:  # 至少像一条指令
            lines.append(line)
    return lines


class SelfInstructPipeline:
    """
    Self-Instruct 流水线：从种子指令 + 已生成指令中抽样，引导模型生成新指令，过滤后写入输出。
    """

    def __init__(
        self,
        seed_path: str,
        data_output_path: str,
        num_machine_instructions: int = 20,
        human_to_machine_ratio: Tuple[int, int] = (5, 2),
        instruction_filter: Optional[InstructionFilter] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        per_call: int = 5,
    ):
        self.seed_path = seed_path
        self.data_output_path = data_output_path
        self.num_machine_instructions = num_machine_instructions
        self.human_to_machine_ratio = human_to_machine_ratio
        self.instruction_filter = instruction_filter or default_filter_config()
        self.client = _create_client(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.per_call = per_call  # 每次请求生成的条数

    def _sample_seed(self, pool: List[dict], k: int) -> List[dict]:
        return random.sample(pool, min(k, len(pool)))

    def _sample_machine(self, pool: List[dict], k: int) -> List[dict]:
        return random.sample(pool, min(k, len(pool))) if pool else []

    def _write_append(self, items: List[dict]) -> None:
        Path(self.data_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_output_path, "a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def generate(self) -> str:
        """执行生成，返回输出文件路径。"""
        seed_pool = load_seed_instructions(self.seed_path)
        if not seed_pool:
            raise ValueError(f"种子文件为空或格式有误: {self.seed_path}" + SEED_FORMAT_HINT)

        # 若输出文件已存在，可从中加载已有机器指令用于抽样和相似度过滤
        existing_path = self.data_output_path
        machine_pool: List[dict] = []
        if Path(existing_path).exists():
            for line in open(existing_path, "r", encoding="utf-8"):
                line = line.strip()
                if line:
                    try:
                        machine_pool.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        else:
            Path(existing_path).parent.mkdir(parents=True, exist_ok=True)

        num_human, num_machine = self.human_to_machine_ratio
        target = self.num_machine_instructions
        existing_instructions = [d["instruction"] for d in machine_pool]
        added = 0
        round_no = 0

        while added < target:
            round_no += 1
            human_samples = self._sample_seed(seed_pool, num_human)
            machine_samples = self._sample_machine(machine_pool, num_machine)
            examples = "\n".join(
                [s["instruction"] for s in (human_samples + machine_samples)]
            )
            to_gen = min(self.per_call, target - added)
            prompt = GENERATION_PROMPT.format(num=to_gen, examples=examples)

            try:
                text = _call_llm(
                    self.client,
                    prompt,
                    model=self.model,
                    temperature=self.temperature,
                )
            except Exception as e:
                print(f"[Round {round_no}] LLM 调用失败: {e}")
                break

            new_instructions = parse_generated_lines(text)
            new_items = []
            for i, instr in enumerate(new_instructions):
                if not self.instruction_filter.apply(instr):
                    continue
                if instr in existing_instructions:
                    continue
                existing_instructions.append(instr)
                new_items.append({
                    "instruction": instr,
                    "input": "",
                    "output": "",
                })
                machine_pool.append(new_items[-1])
                added += 1
                if added >= target:
                    break

            if new_items:
                self._write_append(new_items)
                print(f"[Round {round_no}] 新增 {len(new_items)} 条，累计 {added}/{target}")

        return self.data_output_path


def run_cli(
    seed_path: str = "seed_tasks.jsonl",
    output_path: str = "output/self_instruct_output.jsonl",
    num_instructions: int = 20,
    human_ratio: int = 5,
    machine_ratio: int = 2,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
):
    """命令行式运行：在 self_instruct 目录下执行时可用默认路径。"""
    pipeline = SelfInstructPipeline(
        seed_path=seed_path,
        data_output_path=output_path,
        num_machine_instructions=num_instructions,
        human_to_machine_ratio=(human_ratio, machine_ratio),
        model=model,
        temperature=temperature,
    )
    out = pipeline.generate()
    print(f"生成完成，输出: {out}")
    return out
