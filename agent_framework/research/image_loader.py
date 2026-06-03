"""用户上传图片：视觉 caption + 本地向量索引（对齐 PDF 加载模式）。"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from agent_framework.research.vector_index import (
    MAX_VECTOR_RESULTS,
    build_vector_index_from_texts,
    search_vector_index,
)

# 注入主控 prompt 的单图/总长度上限（字符）
MAX_CHARS_PER_IMAGE_BRIEF = 2800
MAX_CHARS_TOTAL_BRIEF = 10000
MAX_CHARS_TOOL_RESPONSE = 8000
MAX_IMAGE_BYTES = int(os.getenv("IMAGE_MAX_BYTES", str(8 * 1024 * 1024)))

_IMAGE_STORE: dict[str, dict[str, Any]] = {}
_IMAGE_VECTOR_INDEX: list[dict] = []

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _caption_on_load_default() -> bool:
    return os.getenv("IMAGE_CAPTION_ON_LOAD", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def get_image_store() -> dict[str, dict[str, Any]]:
    return _IMAGE_STORE


def get_image_vector_index() -> list[dict]:
    return _IMAGE_VECTOR_INDEX


def set_image_store(store: dict[str, dict[str, Any]]) -> None:
    global _IMAGE_STORE
    _IMAGE_STORE = dict(store)


def set_image_vector_index(index: list[dict]) -> None:
    global _IMAGE_VECTOR_INDEX
    _IMAGE_VECTOR_INDEX = list(index)


def clear_image_store() -> None:
    set_image_store({})
    set_image_vector_index([])


def image_store_summary() -> str:
    if not _IMAGE_STORE:
        return "图片库：未加载"
    names = sorted(_IMAGE_STORE)
    captioned = sum(1 for n in names if (_IMAGE_STORE[n].get("description") or "").strip())
    return f"图片库：{len(names)} 张（已解读 {captioned} 张）；文件：{', '.join(names)}"


def image_vector_index_summary() -> str:
    index = get_image_vector_index()
    if not index:
        if not _IMAGE_STORE:
            return "图片向量索引：未建立（无已登记图片）"
        return "图片向量索引：未建立（将在 search_loaded_image_vectors / read_loaded_image 时生成 caption 并建索引）"
    images = sorted({item["image"] for item in index})
    return f"图片向量索引：{len(images)} 张图片，{len(index)} 个 caption 文本块；文件：{', '.join(images)}"


def rebuild_image_vector_index() -> list[dict]:
    """根据已生成的视觉解读（caption）重建稀疏向量索引。"""
    texts: dict[str, str] = {}
    for name, record in _IMAGE_STORE.items():
        desc = (record.get("description") or "").strip()
        if desc:
            texts[name] = desc
    index = build_vector_index_from_texts(texts, source_key="image")
    set_image_vector_index(index)
    return index


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n\n…（已截断，全文共 {len(text)} 字符，可用 read_loaded_image / search_loaded_image_vectors 继续查看）"
    )


def _find_sections(text: str, hint: str, max_chars: int) -> str:
    if not hint.strip():
        return _truncate(text, max_chars)
    hint_lower = hint.lower()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    matched = [p for p in paragraphs if hint_lower in p.lower()]
    if not matched:
        matched = [p for p in paragraphs if any(w in p.lower() for w in hint_lower.split())]
    if matched:
        return _truncate("\n\n".join(matched[:16]), max_chars)
    return _truncate(text, max_chars)


def _mime_for_path(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed and guessed.startswith("image/"):
        return guessed
    ext = path.suffix.lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }
    return mapping.get(ext, "image/png")


def _image_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"图片过大（{len(data)} 字节 > {MAX_IMAGE_BYTES}），请压缩后重试：{path.name}"
        )
    mime = _mime_for_path(path)
    b64 = base64.standard_b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _describe_image_with_vision(path: Path, *, user_task_hint: str = "") -> str:
    """调用视觉模型生成 caption（需配置 VISION_* 或支持图像的 LLM）。"""
    from agent_framework.llm import create_vision_llm

    llm = create_vision_llm()
    task = (user_task_hint or "语音鉴伪 / 深度伪造 / 模型改进研究").strip()[:800]
    prompt = (
        "你是科研助手。请详细解读这张用户上传的图片，供后续 AI Agent 检索与引用（不要寒暄）。\n"
        "按以下结构输出中文 Markdown：\n"
        "## 图片类型\n"
        "## 可见文字与数字\n"
        "## 图表/架构/流程要点\n"
        "## 与语音鉴伪或模型改进的关联\n"
        "## 待核实问题\n"
        f"\n用户任务背景（仅供参考）：{task}\n"
        "若看不清某处，明确写「无法辨认」，不要编造。"
    )
    data_url = _image_to_data_url(path)
    response = llm.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            )
        ]
    )
    text = (response.content or "").strip()
    if not text:
        raise RuntimeError("视觉模型返回为空")
    return text


def _fallback_description(path: Path, *, error: str) -> str:
    stat = path.stat()
    return (
        f"（未能自动解读图片：{error}）\n"
        f"- 文件名：{path.name}\n"
        f"- 路径：{path}\n"
        f"- 大小：{stat.st_size} 字节\n"
        "请在 .env 配置 VISION_API_KEY / VISION_MODEL（如 gpt-4o、qwen-vl-plus 等支持图像的模型），"
        "或改用 PDF/文字说明。"
    )


def ensure_image_caption(
    image_filename: str,
    *,
    user_task_hint: str = "",
) -> str:
    """懒加载：确保该图片已有视觉 caption，并更新向量索引。"""
    store = get_image_store()
    key = image_filename.strip()
    if key not in store:
        matches = [k for k in store if key.lower() in k.lower()]
        if len(matches) == 1:
            key = matches[0]
        else:
            raise KeyError(f"未找到图片：{image_filename}")

    existing = (store[key].get("description") or "").strip()
    if existing:
        return existing

    path = Path(store[key]["path"])
    try:
        description = _describe_image_with_vision(path, user_task_hint=user_task_hint)
    except Exception as e:
        description = _fallback_description(path, error=str(e))

    store[key]["description"] = description
    rebuild_image_vector_index()
    return description


def ensure_all_image_captions(*, user_task_hint: str = "") -> None:
    """为所有已登记、尚未解读的图片生成 caption。"""
    for name in sorted(get_image_store()):
        if not (get_image_store()[name].get("description") or "").strip():
            ensure_image_caption(name, user_task_hint=user_task_hint)


def search_image_vectors(
    query: str,
    *,
    user_task_hint: str = "",
    top_k: int = MAX_VECTOR_RESULTS,
    max_chars: int = 5000,
) -> str:
    """在图片 caption 的本地向量索引中检索相关片段。"""
    if not get_image_store():
        return "当前未加载任何图片。请在 input.md 使用 `IMAGE: 路径` 或 `--image` 加载。"

    ensure_all_image_captions(user_task_hint=user_task_hint)
    index = get_image_vector_index()
    if not index:
        return "图片 caption 为空，无法建立向量索引。请检查 VISION_* 配置后重试。"

    result = search_vector_index(
        index,
        query,
        source_key="image",
        top_k=top_k,
        max_chars=max_chars,
        header="## 图片 caption 向量检索结果",
        empty_hint="图片向量索引为空。",
    )
    if "未检索到相关片段" in result:
        return f"{result}\n（已检索全部已解读图片的 caption 文本块）"
    return result


def load_user_images(
    paths: list[str | Path],
    *,
    user_task_hint: str = "",
    caption_on_load: bool | None = None,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """
    登记用户图片；默认懒加载 caption（首次检索/阅读时再调视觉模型）。

    Returns:
        (供主控使用的摘要上下文, 图片记录字典)
    """
    if caption_on_load is None:
        caption_on_load = _caption_on_load_default()

    store: dict[str, dict[str, Any]] = {}
    brief_parts: list[str] = []
    total_brief = 0

    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"图片不存在：{path}")
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"不支持的图片格式：{path}（支持 {', '.join(sorted(IMAGE_EXTENSIONS))}）"
            )

        name = path.name
        description = ""
        if caption_on_load:
            try:
                description = _describe_image_with_vision(path, user_task_hint=user_task_hint)
            except Exception as e:
                description = _fallback_description(path, error=str(e))

        store[name] = {
            "path": str(path),
            "description": description,
            "mime": _mime_for_path(path),
            "size_bytes": path.stat().st_size,
        }

        if description:
            per_limit = min(
                MAX_CHARS_PER_IMAGE_BRIEF,
                max(1500, MAX_CHARS_TOTAL_BRIEF // max(len(paths), 1)),
            )
            excerpt = _truncate(description, per_limit)
            block = f"### 图片：{name}\n（路径：{path}）\n\n{excerpt}"
            if total_brief + len(block) > MAX_CHARS_TOTAL_BRIEF:
                remain = MAX_CHARS_TOTAL_BRIEF - total_brief
                if remain > 400:
                    brief_parts.append(_truncate(block, remain))
                break
            brief_parts.append(block)
            total_brief += len(block)

    set_image_store(store)
    if caption_on_load:
        rebuild_image_vector_index()

    if not store:
        return "", store

    if brief_parts:
        header = (
            f"## 用户提供的参考图片（共 {len(store)} 张）\n"
            "以下为视觉解读摘要；请优先 `search_loaded_image_vectors` 定位相关片段，"
            "再用 `read_loaded_image` 精读。\n\n"
        )
        return header + "\n\n".join(brief_parts), store

    names = ", ".join(sorted(store))
    header = (
        f"## 用户提供的参考图片（共 {len(store)} 张）\n"
        f"已登记：{names}。\n"
        "caption 与向量索引将在首次调用 `search_loaded_image_vectors` 或 `read_loaded_image` 时生成。\n"
        "子 Agent 请先向量检索定位相关图片，再按需精读；勿臆测图表内容。"
    )
    return header, store


def read_image_from_store(
    image_filename: str,
    focus_hint: str = "",
    *,
    user_task_hint: str = "",
) -> str:
    """供工具调用：确保 caption 后按 focus 返回解读文本。"""
    store = get_image_store()
    if not store:
        return "当前未加载任何图片。请在 input.md 使用 `IMAGE: 路径` 或 `--image` 加载。"

    key = image_filename.strip()
    if key not in store:
        matches = [k for k in store if key.lower() in k.lower()]
        if len(matches) == 1:
            key = matches[0]
        elif matches:
            return f"未精确匹配。可选文件名：{', '.join(matches)}"
        else:
            return f"未找到「{image_filename}」。已加载：{', '.join(store.keys())}"

    try:
        desc = ensure_image_caption(key, user_task_hint=user_task_hint)
    except KeyError as e:
        return str(e)

    body = _find_sections(desc, focus_hint, MAX_CHARS_TOOL_RESPONSE)
    return f"## {key}\n（路径：{store[key].get('path', '')}）\n\n{body}"


def list_loaded_images_brief() -> str:
    store = get_image_store()
    if not store:
        return "当前未加载图片。"
    lines = ["## 已加载图片"]
    for name in sorted(store):
        desc = (store[name].get("description") or "").strip()
        if desc:
            preview = _truncate(desc, 400)
        else:
            preview = "（尚未生成视觉解读；调用 search/read 工具时将自动生成）"
        lines.append(f"\n### {name}\n{preview}")
    lines.append(f"\n{image_vector_index_summary()}")
    return "\n".join(lines)


def collect_image_paths(
    image_args: list[str] | None = None,
    images_dir: str | Path | None = None,
) -> list[Path]:
    """合并 --image 显式路径与 images 目录下的图片。"""
    paths: list[Path] = []
    seen: set[str] = set()

    for raw in image_args or []:
        p = Path(raw).expanduser().resolve()
        key = str(p)
        if key not in seen and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
            seen.add(key)

    if images_dir:
        folder = Path(images_dir).expanduser().resolve()
        if folder.is_dir():
            for ext in IMAGE_EXTENSIONS:
                for p in sorted(folder.glob(f"*{ext}")):
                    key = str(p.resolve())
                    if key not in seen:
                        paths.append(p)
                        seen.add(key)

    return paths
