"""用户可见文本清洗与相似度检测。"""

from __future__ import annotations

import re


_PREAMBLE_PATTERNS = (
    r"^好的[，,].*$",
    r"^收到.*$",
    r"^作为.*Agent.*$",
    r"^以下是.*$",
    r"^我已.*$",
    r"^根据审查.*$",
    r"^---\s*$",
)


def normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def is_same_as_brief(followup: str, user_brief: str, threshold: float = 0.82) -> bool:
    """续问是否与首问任务过于相似（误把首问当续问）。"""
    a = normalize_for_compare(followup)
    b = normalize_for_compare(user_brief)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = (a, b) if len(a) < len(b) else (b, a)
    if short in long and len(short) / len(long) > threshold:
        return True
    return False


def plans_nearly_identical(new_plan: str, old_plan: str, threshold: float = 0.92) -> bool:
    a = normalize_for_compare(new_plan)
    b = normalize_for_compare(old_plan)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = (a, b) if len(a) < len(b) else (b, a)
    return short in long and len(short) / len(long) > threshold


def clean_user_markdown(text: str) -> str:
    """去掉寒暄与首个一级标题之前的废话。"""
    if not text:
        return ""
    lines = text.splitlines()
    cleaned: list[str] = []
    seen_heading = False
    for line in lines:
        stripped = line.strip()
        if not seen_heading:
            if stripped.startswith("#"):
                seen_heading = True
                cleaned.append(line)
                continue
            if any(re.match(p, stripped) for p in _PREAMBLE_PATTERNS):
                continue
            if stripped in {"", "---"}:
                continue
            # 跳过「先回答用户追问」之前的说明段（保留该标题本身）
            if "先回答用户追问" in stripped and not stripped.startswith("#"):
                cleaned.append("## 针对用户本轮追问的回答")
                seen_heading = True
                continue
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    if not result.startswith("#"):
        m = re.search(r"^#\s+.+", result, re.MULTILINE)
        if m:
            result = result[m.start() :].strip()
    return result
