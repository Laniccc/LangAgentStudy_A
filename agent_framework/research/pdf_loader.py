"""本地论文 PDF 加载与文本提取。"""

from __future__ import annotations

from pathlib import Path

# 运行时论文库：{文件名: 全文}
_PAPER_STORE: dict[str, str] = {}

# 注入主控 prompt 时单篇/总长度上限（字符）
MAX_CHARS_PER_PAPER_BRIEF = 6000
MAX_CHARS_TOTAL_BRIEF = 18000
# 工具单次返回上限
MAX_CHARS_TOOL_RESPONSE = 12000


def get_paper_store() -> dict[str, str]:
    return _PAPER_STORE


def set_paper_store(store: dict[str, str]) -> None:
    global _PAPER_STORE
    _PAPER_STORE = dict(store)


def clear_paper_store() -> None:
    set_paper_store({})


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("请安装 pypdf：pip install pypdf") from e

    reader = PdfReader(str(path))
    parts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    if not parts:
        return f"（未能从 {path.name} 提取文本，可能为扫描版 PDF，需 OCR）"
    return "\n\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n…（已截断，全文共 {len(text)} 字符，可用 read_loaded_paper 工具继续读取）"


def _find_sections(text: str, hint: str, max_chars: int) -> str:
    """按关键词抽取相关段落。"""
    if not hint.strip():
        return _truncate(text, max_chars)

    hint_lower = hint.lower()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    matched = [p for p in paragraphs if hint_lower in p.lower()]
    if not matched:
        matched = [p for p in paragraphs if any(w in p.lower() for w in hint_lower.split())]

    if matched:
        joined = "\n\n".join(matched[:20])
        return _truncate(joined, max_chars)

    return _truncate(text, max_chars)


def load_paper_pdfs(paths: list[str | Path]) -> tuple[str, dict[str, str]]:
    """
    加载一个或多个 PDF，写入运行时论文库。

    Returns:
        (供主控使用的摘要上下文, 论文字典)
    """
    store: dict[str, str] = {}
    brief_parts: list[str] = []
    total_brief = 0

    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF 不存在：{path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"非 PDF 文件：{path}")

        text = _extract_pdf_text(path)
        name = path.name
        store[name] = text

        per_limit = min(
            MAX_CHARS_PER_PAPER_BRIEF,
            max(2000, MAX_CHARS_TOTAL_BRIEF // max(len(paths), 1)),
        )
        excerpt = _truncate(text, per_limit)
        block = f"### 论文：{name}\n（路径：{path}）\n\n{excerpt}"
        if total_brief + len(block) > MAX_CHARS_TOTAL_BRIEF:
            remain = MAX_CHARS_TOTAL_BRIEF - total_brief
            if remain > 500:
                brief_parts.append(_truncate(block, remain))
            break
        brief_parts.append(block)
        total_brief += len(block)

    set_paper_store(store)
    if not brief_parts:
        return "", store

    header = (
        f"## 用户提供的参考论文（共 {len(store)} 篇）\n"
        "以下为摘要节选；子 Agent 可通过 `read_loaded_paper` 工具读取全文或按章节检索。\n\n"
    )
    return header + "\n\n".join(brief_parts), store


def read_paper_from_store(paper_filename: str, section_hint: str = "") -> str:
    """供工具调用的论文读取逻辑。"""
    store = get_paper_store()
    if not store:
        return "当前未加载任何 PDF。请使用 --pdf 或在 data/papers/ 目录放置论文后重试。"

    key = paper_filename.strip()
    if key not in store:
        # 模糊匹配
        matches = [k for k in store if key.lower() in k.lower()]
        if len(matches) == 1:
            key = matches[0]
        elif matches:
            return f"未精确匹配。可选文件名：{', '.join(matches)}"
        else:
            return f"未找到「{paper_filename}」。已加载：{', '.join(store.keys())}"

    text = store[key]
    return f"## {key}\n\n" + _find_sections(text, section_hint, MAX_CHARS_TOOL_RESPONSE)
