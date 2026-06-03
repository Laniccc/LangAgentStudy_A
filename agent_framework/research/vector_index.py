"""PDF / 图片共用的本地稀疏向量索引工具。"""

from __future__ import annotations

import math
import re
from collections import Counter

CHUNK_CHARS = 1400
CHUNK_OVERLAP = 220
MAX_VECTOR_RESULTS = 5


def tokenize_for_vector(text: str) -> list[str]:
    raw = (text or "").lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9_\-+/\.]{1,}|[\u4e00-\u9fff]{2,}", raw)
    cjk_bigrams: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            cjk_bigrams.extend(token[i : i + 2] for i in range(max(len(token) - 1, 0)))
    return tokens + cjk_bigrams


def vectorize_text(text: str) -> dict[str, float]:
    counts = Counter(tokenize_for_vector(text))
    if not counts:
        return {}
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


def cosine_sparse(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())


def chunk_text(
    text: str,
    *,
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    cleaned = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    step = max(chunk_chars - overlap, 1)
    while start < len(cleaned):
        end = min(start + chunk_chars, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start += step
    return chunks


def build_vector_index_from_texts(
    items: dict[str, str],
    *,
    source_key: str = "source",
) -> list[dict]:
    """items: {文件名: 全文或 caption} → 索引条目列表。"""
    index: list[dict] = []
    for filename in sorted(items):
        for idx, chunk in enumerate(chunk_text(items[filename])):
            vector = vectorize_text(chunk)
            if not vector:
                continue
            index.append(
                {
                    source_key: filename,
                    "chunk_id": idx,
                    "text": chunk,
                    "vector": vector,
                }
            )
    return index


def search_vector_index(
    index: list[dict],
    query: str,
    *,
    source_key: str = "source",
    top_k: int = MAX_VECTOR_RESULTS,
    max_chars: int = 5000,
    header: str = "## 向量检索结果",
    empty_hint: str = "索引为空",
) -> str:
    if not index:
        return empty_hint

    query_vector = vectorize_text(query)
    if not query_vector:
        return "查询为空或无法向量化，请提供更具体的关键词。"

    scored = [(cosine_sparse(query_vector, item["vector"]), item) for item in index]
    scored = [(score, item) for score, item in scored if score > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return f"未检索到相关片段。查询：{query}"

    parts = [f"{header}\n查询：{query}"]
    total = len(parts[0])
    for rank, (score, item) in enumerate(scored[: max(top_k, 1)], 1):
        src = item.get(source_key, "")
        block = (
            f"\n\n### {rank}. {src} · chunk {item['chunk_id']} · score={score:.3f}\n"
            f"{item['text']}"
        )
        if total + len(block) > max_chars:
            remain = max_chars - total
            if remain > 200:
                parts.append(block[:remain] + "\n...（已截断）")
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)
