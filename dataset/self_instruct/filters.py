# dataset/self_instruct/filters.py
"""
Self-Instruct 指令过滤：长度、关键词、标点、语言、相似度等。
FilterFunction.apply(instruction: str) -> True 保留 / False 过滤。
"""
import re
from abc import ABC, abstractmethod
from typing import List, Optional


class FilterFunction(ABC):
    @abstractmethod
    def apply(self, instruction: str) -> bool:
        """返回 True 表示保留该指令，False 表示过滤掉。"""
        pass


class LengthFilter(FilterFunction):
    """过滤长度不在 [min_len, max_len] 之间的指令。"""

    def __init__(self, min_len: int = 5, max_len: int = 500):
        self.min_len = min_len
        self.max_len = max_len

    def apply(self, instruction: str) -> bool:
        n = len(instruction.strip())
        return self.min_len <= n <= self.max_len


class KeywordFilter(FilterFunction):
    """过滤包含指定关键词的指令（如敏感词、禁止话题）。"""

    def __init__(self, keywords: List[str], exclude: bool = True):
        self.keywords = [k.lower() for k in keywords]
        self.exclude = exclude  # True=包含则过滤，False=不包含则过滤

    def apply(self, instruction: str) -> bool:
        lower = instruction.lower()
        has_any = any(k in lower for k in self.keywords)
        return (not has_any) if self.exclude else has_any


class PunctuationFilter(FilterFunction):
    """过滤以非字母数字开头的指令。"""

    def apply(self, instruction: str) -> bool:
        s = instruction.strip()
        if not s:
            return False
        return s[0].isalnum() or s[0].isalpha()


class NonEnglishFilter(FilterFunction):
    """过滤不以英文字母开头的指令（若需纯英文指令集可用）。"""

    def apply(self, instruction: str) -> bool:
        s = instruction.strip()
        if not s:
            return False
        return s[0].isascii() and s[0].isalpha()


class RougeSimilarityFilter(FilterFunction):
    """基于简单 n-gram 重叠过滤与已有指令过于相似的指令。"""

    def __init__(self, existing_instructions: List[str], threshold: float = 0.6, n: int = 2):
        self.existing = existing_instructions
        self.threshold = threshold
        self.n = n

    def _ngrams(self, text: str) -> set:
        text = text.strip().lower()
        return set(text[i : i + self.n] for i in range(len(text) - self.n + 1))

    def apply(self, instruction: str) -> bool:
        if not self.existing:
            return True
        ng = self._ngrams(instruction)
        if not ng:
            return True
        for ex in self.existing:
            ex_ng = self._ngrams(ex)
            if ex_ng:
                overlap = len(ng & ex_ng) / len(ng)
                if overlap >= self.threshold:
                    return False
        return True


class InstructionFilter:
    """组合多个 FilterFunction，全部通过才保留。"""

    def __init__(self, filters: Optional[List[FilterFunction]] = None):
        self.filters = filters or []

    def add(self, f: FilterFunction) -> "InstructionFilter":
        self.filters.append(f)
        return self

    def apply(self, instruction: str) -> bool:
        return all(f.apply(instruction) for f in self.filters)


def default_filter_config(
    min_len: int = 5,
    max_len: int = 200,
    keywords: Optional[List[str]] = None,
    existing_instructions: Optional[List[str]] = None,
    similarity_threshold: float = 0.7,
) -> InstructionFilter:
    """常用默认过滤组合。"""
    keywords = keywords or ["图像", "视频", "image", "video"]
    inst = InstructionFilter()
    inst.add(LengthFilter(min_len=min_len, max_len=max_len))
    inst.add(KeywordFilter(keywords=keywords))
    inst.add(PunctuationFilter())
    if existing_instructions is not None:
        inst.add(RougeSimilarityFilter(existing_instructions, threshold=similarity_threshold))
    return inst
