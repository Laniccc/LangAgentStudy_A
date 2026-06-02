"""会话级结构化记忆与本地向量检索。"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_framework.research.session import get_runs_dir

MEMORY_TEXT_LIMIT = 2400
MEMORY_VIEW_LIMIT = 3200

ROLE_TYPE_PREFERENCES: dict[str, tuple[str, ...]] = {
    "prompt_agent": ("user_intent", "decision", "plan_summary", "rejected_option"),
    "orchestrator": ("decision", "plan_summary", "user_intent", "review_issue"),
    "sub_agent": ("evidence", "research_result", "decision", "user_intent"),
    "reviewer": ("review_issue", "decision", "evidence", "user_intent"),
}


def _memory_dir(session_id: str) -> Path:
    path = get_runs_dir() / "store" / session_id / "memory"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _memory_path(session_id: str) -> Path:
    return _memory_dir(session_id) / "items.jsonl"


def _tokenize(text: str) -> list[str]:
    raw = (text or "").lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9_\-+/\.]{1,}|[\u4e00-\u9fff]{2,}", raw)
    cjk_bigrams: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", token):
            cjk_bigrams.extend(token[i : i + 2] for i in range(max(len(token) - 1, 0)))
    return tokens + cjk_bigrams


def _vectorize(text: str) -> dict[str, float]:
    counts = Counter(_tokenize(text))
    if not counts:
        return {}
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())


def _normalize_tags(tags: list[str] | tuple[str, ...] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags or []:
        if tag is None:
            continue
        value = str(tag).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out[:12]


def _extract_tags(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-+/\.]{2,}|[\u4e00-\u9fff]{2,}", text or "")
    stop = {"the", "and", "for", "with", "that", "this", "from", "into"}
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key in stop or key in seen:
            continue
        seen.add(key)
        out.append(token)
        if len(out) >= 8:
            break
    return out


def _clip(text: str, limit: int = MEMORY_TEXT_LIMIT) -> str:
    raw = (text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "\n...(truncated)"


def append_structured_memory(
    session_id: str,
    *,
    memory_type: str,
    text: str,
    source: str,
    tags: list[str] | tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """写入一条可检索的结构化记忆。"""
    if not session_id:
        return
    clipped = _clip(text)
    if not clipped:
        return
    merged_tags = _normalize_tags([*(tags or []), *_extract_tags(clipped)])
    item = {
        "id": uuid.uuid4().hex[:12],
        "type": memory_type,
        "source": source,
        "text": clipped,
        "tags": merged_tags,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vector": _vectorize(" ".join([memory_type, source, " ".join(merged_tags), clipped])),
    }
    path = _memory_path(session_id)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_memory_items(session_id: str, *, limit: int = 400) -> list[dict[str, Any]]:
    path = _memory_path(session_id)
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines[-limit:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("text"):
            items.append(item)
    return items


def retrieve_memory_items(
    session_id: str,
    *,
    query: str,
    role: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """按当前查询和 Agent 角色召回历史记忆。"""
    if not session_id or not query:
        return []
    items = load_memory_items(session_id)
    if not items:
        return []
    query_vector = _vectorize(query)
    preferred = ROLE_TYPE_PREFERENCES.get(role, ())
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, item in enumerate(items):
        vector = item.get("vector") or {}
        score = _cosine(query_vector, vector)
        if item.get("type") in preferred:
            score += 0.08
        if role in (item.get("metadata") or {}).get("roles", []):
            score += 0.05
        if score <= 0:
            continue
        scored.append((score, idx, item))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [item for _, _, item in scored[: max(top_k, 1)]]


def render_memory_view(items: list[dict[str, Any]], *, max_chars: int = MEMORY_VIEW_LIMIT) -> str:
    if not items:
        return ""
    parts = ["## 相关历史记忆（向量检索召回）"]
    total = len(parts[0])
    for item in items:
        tags = ", ".join(item.get("tags") or [])
        block = (
            f"\n\n### {item.get('type', 'memory')} · {item.get('source', '')}\n"
            f"tags: {tags}\n"
            f"{item.get('text', '')}"
        )
        if total + len(block) > max_chars:
            remain = max_chars - total
            if remain > 160:
                parts.append(block[:remain] + "\n...(truncated)")
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


def retrieve_memory_view(
    session_id: str,
    *,
    query: str,
    role: str,
    top_k: int = 5,
    max_chars: int = MEMORY_VIEW_LIMIT,
) -> str:
    return render_memory_view(
        retrieve_memory_items(session_id, query=query, role=role, top_k=top_k),
        max_chars=max_chars,
    )


def extract_plan_modules(plan_text: str, *, limit: int = 12) -> list[str]:
    """从方案标题/小节中提取可被用户追问指代的模块名。"""
    modules: list[str] = []
    seen: set[str] = set()
    for line in (plan_text or "").splitlines():
        stripped = line.strip(" #*-")
        if not stripped:
            continue
        if any(key in stripped for key in ("P0", "P1", "P2", "前端", "模块", "融合", "注意力", "小波")):
            value = stripped[:120]
            if value not in seen:
                seen.add(value)
                modules.append(value)
        if len(modules) >= limit:
            break
    return modules
