"""阶段产物文件 Store（借鉴 LangGraph Store 的 phase_outputs 命名空间）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_framework.research.session import get_runs_dir

PHASE_NAMESPACES = (
    "phase_state",
    "phase_outputs",
    "memory",
)


def _session_store_dir(session_id: str) -> Path:
    root = get_runs_dir() / "store" / session_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_phase_output(session_id: str, phase: str, content: str) -> None:
    """写入 phase_outputs/{phase}.md"""
    if not session_id:
        return
    out_dir = _session_store_dir(session_id) / "phase_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{phase}.md"
    path.write_text(content, encoding="utf-8")


def write_phase_state(session_id: str, phase: str, data: dict[str, Any]) -> None:
    """写入 phase_state/{phase}.json"""
    if not session_id:
        return
    out_dir = _session_store_dir(session_id) / "phase_state"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{phase}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_memory(session_id: str, entry: dict[str, Any]) -> None:
    """追加会话级 memory 条目（用户追问、审批记录等）。"""
    if not session_id:
        return
    mem_dir = _session_store_dir(session_id) / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    path = mem_dir / "log.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def sync_from_state_patch(session_id: str, patch: dict) -> None:
    """将节点返回的状态补丁同步到文件 Store。"""
    if not session_id:
        return
    artifacts = patch.get("phase_artifacts") or {}
    for phase, text in artifacts.items():
        if text:
            write_phase_output(session_id, phase, text)
    status = patch.get("phase_status") or {}
    for phase, st in status.items():
        write_phase_state(session_id, phase, {"status": st})


def load_phase_output(session_id: str, phase: str) -> str | None:
    path = _session_store_dir(session_id) / "phase_outputs" / f"{phase}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None
