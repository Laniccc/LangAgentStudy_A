"""本地论文 PDF 加载与文本提取。"""

from __future__ import annotations

from pathlib import Path

from agent_framework.research.vector_index import (
    MAX_VECTOR_RESULTS,
    build_vector_index_from_texts,
    search_vector_index,
)

# 运行时论文库：{文件名: 全文}
_PAPER_STORE: dict[str, str] = {}
_PAPER_VECTOR_INDEX: list[dict] = []

# 注入主控 prompt 时单篇/总长度上限（字符）
MAX_CHARS_PER_PAPER_BRIEF = 6000
MAX_CHARS_TOTAL_BRIEF = 18000
# 工具单次返回上限
MAX_CHARS_TOOL_RESPONSE = 12000


def get_paper_store() -> dict[str, str]:
    return _PAPER_STORE


def get_paper_vector_index() -> list[dict]:
    return _PAPER_VECTOR_INDEX


def set_paper_store(store: dict[str, str]) -> None:
    global _PAPER_STORE
    _PAPER_STORE = dict(store)


def set_paper_vector_index(index: list[dict]) -> None:
    global _PAPER_VECTOR_INDEX
    _PAPER_VECTOR_INDEX = list(index)


def clear_paper_store() -> None:
    set_paper_store({})
    set_paper_vector_index([])


def build_paper_vector_index(store: dict[str, str]) -> list[dict]:
    """为已加载 PDF 建立本地稀疏向量索引，供主控/工具按需检索。"""
    return build_vector_index_from_texts(store, source_key="paper")


def search_paper_vectors(query: str, *, top_k: int = MAX_VECTOR_RESULTS, max_chars: int = 5000) -> str:
    """从本地 PDF 向量索引中检索相关片段。"""
    index = get_paper_vector_index()
    if not index:
        return "当前未建立 PDF 向量索引。请先通过 --pdf 或 data/papers 加载论文。"

    return search_vector_index(
        index,
        query,
        source_key="paper",
        top_k=top_k,
        max_chars=max_chars,
        header="## PDF 向量检索结果",
        empty_hint="当前未建立 PDF 向量索引。",
    )


def paper_vector_index_summary() -> str:
    index = get_paper_vector_index()
    if not index:
        return "PDF 向量索引：未建立"
    papers = sorted({item["paper"] for item in index})
    return f"PDF 向量索引：{len(papers)} 篇论文，{len(index)} 个文本块；文件：{', '.join(papers)}"


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
    set_paper_vector_index(build_paper_vector_index(store))
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
