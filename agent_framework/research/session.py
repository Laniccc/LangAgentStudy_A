"""研究会话：Checkpointer、thread_id、Run 元数据。"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

RUNS_DIR = Path(".runs")
CHECKPOINT_DB = RUNS_DIR / "checkpoints.sqlite"
META_DIR = RUNS_DIR / "sessions"

_checkpointer = None


def get_runs_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR


def get_checkpointer():
    """优先 SqliteSaver（跨进程/重启可续），否则 MemorySaver。"""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    get_runs_dir()
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
        _checkpointer = SqliteSaver(conn)
    except ImportError:
        _checkpointer = MemorySaver()
    return _checkpointer


def new_thread_id() -> str:
    return f"research-{uuid.uuid4().hex[:12]}"


def normalize_thread_id(thread_id: str | None, *, mode: str = "research") -> str | None:
    """
    规范化 thread_id。用户常只输入 hex 后缀（如 24b9f385c515），自动补 research- 前缀。
    """
    if not thread_id:
        return None
    tid = thread_id.strip()
    if tid.startswith(("research-", "chat-")):
        return tid
    hex_part = tid.removeprefix("research-").removeprefix("chat-")
    if len(hex_part) == 12 and all(c in "0123456789abcdef" for c in hex_part.lower()):
        prefix = "chat-" if mode == "chat" else "research-"
        return f"{prefix}{hex_part.lower()}"
    return tid


def make_thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def save_session_meta(thread_id: str, meta: dict[str, Any]) -> None:
    get_runs_dir()
    path = META_DIR / f"{thread_id}.json"
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing.update(meta)
    existing["thread_id"] = thread_id
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session_meta(thread_id: str) -> dict[str, Any]:
    path = META_DIR / f"{thread_id}.json"
    if not path.exists():
        return {"thread_id": thread_id}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"thread_id": thread_id}


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    get_runs_dir()
    items: list[dict[str, Any]] = []
    for path in sorted(META_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append(data)
        except (json.JSONDecodeError, OSError):
            continue
        if len(items) >= limit:
            break
    return items
