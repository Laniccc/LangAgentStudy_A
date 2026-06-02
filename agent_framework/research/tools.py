"""语音鉴伪研究专用工具。

工具分类（借鉴 Phase / Control / Utility）：
- **Phase**：产出调研与知识内容（检索、读 PDF、列指标）
- **Utility**：连接检查等横切能力（见 scripts/check_api_connection.py）
- **Control**：流程控制由图节点与 CLI 完成（人工审批、跳过补充、续问），非 LLM 工具
"""

from enum import Enum

import json
import warnings
from pathlib import Path

from langchain_core.tools import tool

from agent_framework.research.pdf_loader import read_paper_from_store

class ToolCategory(str, Enum):
    PHASE = "phase"
    CONTROL = "control"
    UTILITY = "utility"


# 核心优化目标（工具与知识库默认对齐）
PRIMARY_BENCHMARKS = ("ASVspoof 2021 LA", "ASVspoof 2021 DF")
PRIMARY_METRIC = "EER"

# 领域知识库（开源可达成果摘要，供检索失败时回退）
_ANTISPOOFING_KB = {
    "primary_objective": {
        "目标": f"降低模型在 {PRIMARY_BENCHMARKS[0]}、{PRIMARY_BENCHMARKS[1]} 上的 {PRIMARY_METRIC}",
        "LA 场景": "逻辑访问（Logic Access）；开发集调参、评估集报 EER；关注 codec/synthesis 攻击",
        "DF 场景": "深度伪造（Deepfake）；跨说话人/跨域；LA 上训练、DF 上测试是常见泛化设定",
        "关联检索词": "ASVspoof 2021 LA EER, ASVspoof 2021 DF EER, anti-spoofing, audio deepfake, deepfake detection",
    },
    "benchmarks": {
        "ASVspoof 2021 LA": "逻辑访问；核心指标 EER、min t-DCF；具体 SOTA 数值须以联网检索为准（勿用本库作榜单）",
        "ASVspoof 2021 DF": "深度伪造音频；强调未知攻击与跨域；与 LA 联合报告是论文常规",
        "ASVspoof 2019 LA": "前代 LA 基线；可与 2021 LA 对照迁移",
        "ASVspoof 5": "新一代多场景评测趋势",
        "ADD 2022/2023": "音频深度伪造检测挑战赛；deepfake 关键词常出现",
    },
    "models": {
        "AASIST": "历史强基线之一；具体 EER 与是否仍为 SOTA 须检索 2023–2026 文献",
        "RawNet2/3": "经典端到端 Raw waveform CNN；对比时须注明文献年份",
        "WavLM/SSL-front": "自监督前端 + 后端分类器；降 EER 常见改动点",
        "LFCC-LCNN": "经典手工特征基线",
        "FAD": "融合辅助任务、多尺度建模",
    },
    "techniques": {
        "frontend": "Raw waveform、LEAF、SincNet、Wav2Vec/WavLM 特征",
        "pooling": "Attentive statistics pooling、ASTP",
        "augmentation": "RawBoost、MUSAN noise、codec 模拟、混响",
        "loss": "A-Softmax、AM-Softmax、二分类 CE + 对比学习",
        "generalization": "DG、MMD、对抗训练、LA→DF 跨域",
    },
    "cross_domain_image": {
        "可迁移方向": "图像 deepfake 检测中的频域/多尺度融合、注意力、对比学习、域泛化、Face Anti-Spoofing",
        "检索建议": "image deepfake detection, face anti-spoofing, transferable to audio spoofing",
        "映射到语音": "时频图≈图像、raw waveform≈1D 信号；CLIP/自监督、MixStyle、DG 方法可对照迁移",
    },
}


def _dedupe_hits(hits: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for h in hits:
        href = h.get("href", "")
        if href and href in seen:
            continue
        if href:
            seen.add(href)
        out.append(h)
    return out


def _format_hits(hits: list[dict], header: str = "") -> str:
    if not hits:
        return ""
    lines = [header] if header else []
    for i, h in enumerate(hits, 1):
        title = h.get("title", "")
        body = (h.get("body") or "")[:320]
        href = h.get("href", "")
        lines.append(f"{i}. **{title}**\n   {body}\n   {href}")
    return "\n\n".join(lines)


def _get_ddgs_class():
    """优先新包 ddgs，回退 duckduckgo_search（屏蔽更名 RuntimeWarning）。"""
    try:
        from ddgs import DDGS

        return DDGS
    except ImportError:
        pass
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*renamed to.*ddgs.*",
                category=RuntimeWarning,
            )
            from duckduckgo_search import DDGS

        return DDGS
    except ImportError:
        return None


def _web_search(queries: list[str], max_per_query: int = 5) -> str:
    """执行多查询联网检索并合并去重。"""
    DDGS = _get_ddgs_class()
    if DDGS is None:
        joined = " | ".join(queries)
        return (
            f"（未安装联网检索库，请执行：pip install ddgs）\n"
            f"建议手动检索：{joined}\n"
            f"关键词参考：ASVspoof 2021 LA EER, ASVspoof 2021 DF EER, "
            f"anti-spoofing, deepfake, audio deepfake, github, arxiv"
        )

    all_hits: list[dict] = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                batch = list(ddgs.text(q, max_results=max_per_query))
                all_hits.extend(batch)
    except Exception as e:
        return f"检索异常：{e}。请换关键词或使用 query_antispoofing_knowledge。"

    merged = _dedupe_hits(all_hits)
    if not merged:
        return f"未检索到结果。已尝试查询：\n" + "\n".join(f"- {q}" for q in queries)

    return _format_hits(merged[:14], header=f"共 {len(merged)} 条结果（已去重）：")


@tool
def query_antispoofing_knowledge(topic: str) -> str:
    """查询语音鉴伪领域知识，默认对齐 ASVspoof 2021 LA/DF 降 EER 目标。

    topic 示例：ASVspoof 2021 LA、DF 跨域、AASIST、图像鉴伪迁移、deepfake
    """
    topic_lower = topic.lower().strip()
    parts = []

    for category, items in _ANTISPOOFING_KB.items():
        if topic_lower in category or category in topic_lower:
            parts.append(f"## {category}\n{json.dumps(items, ensure_ascii=False, indent=2)}")

    for category, items in _ANTISPOOFING_KB.items():
        for key, value in items.items():
            key_l = key.lower()
            val_l = value.lower() if isinstance(value, str) else json.dumps(value, ensure_ascii=False).lower()
            if topic_lower in key_l or topic_lower in val_l:
                parts.append(f"- **{key}** ({category}): {value}")

    disclaimer = (
        "⚠️ **本知识库仅供术语与协议 orientation，不含可信的当前 SOTA/EER。"
        "写入方案的具体数值与创新点必须来自联网检索或 PDF，禁止直接引用下述模型名为「最新 SOTA」。**\n\n"
    )
    if not parts:
        return disclaimer + json.dumps(_ANTISPOOFING_KB, ensure_ascii=False, indent=2)
    return disclaimer + "\n".join(parts)


@tool
def search_asvspoof2021_eer_research(technique_or_model: str) -> str:
    """检索 ASVspoof 2021 LA / DF 上与降低 EER 相关的论文、代码与榜单。

    自动组合 anti-spoofing、deepfake、audio deepfake 等关键词，覆盖 LA 逻辑访问与 DF 深度伪造场景。

    Args:
        technique_or_model: 技术点或模型名，如 AASIST、RawBoost、attention pooling、domain generalization
    """
    t = technique_or_model.strip()
    queries = [
        f"ASVspoof 2021 LA EER {t} 2024 2025 2026 arxiv github",
        f"ASVspoof 2021 DF EER {t} audio deepfake 2024 2025",
        f"ASVspoof 5 {t} anti-spoofing countermeasure",
        f"{t} speech spoofing detection Interspeech ICASSP 2024 2025",
        f"anti-spoofing {t} deepfake audio EER SOTA recent",
    ]
    header = (
        f"## ASVspoof 2021 LA/DF · EER 优化检索\n"
        f"主题：{t} | 目标：降低 LA 与 DF 上的 EER\n\n"
    )
    return header + _web_search(queries)


@tool
def search_deepfake_cross_domain(technique_focus: str) -> str:
    """检索 deepfake / audio deepfake 相关成果，并包含可迁移到语音鉴伪的图像域鉴伪经验。

    覆盖音频深度伪造检测与图像 deepfake、人脸反欺骗（face anti-spoofing）的跨域创新。

    Args:
        technique_focus: 关注点，如 contrastive learning、frequency domain、domain generalization
    """
    t = technique_focus.strip()
    queries = [
        f"audio deepfake detection {t} anti-spoofing EER 2024 2025 2026",
        f"deepfake detection {t} arxiv github 2024 2025",
        f"image deepfake detection {t} transferable audio spoofing",
        f"face anti-spoofing {t} generalization deepfake",
        f"ASVspoof {t} audio deepfake countermeasure",
        f"speech spoofing {t} deepfake neural vocoder detection",
    ]
    header = (
        f"## Deepfake 跨域检索（音频 + 图像可迁移）\n"
        f"主题：{t}\n\n"
    )
    return header + _web_search(queries)


@tool
def search_open_research(query: str) -> str:
    """通用开源研究检索（自动增强 ASVspoof 2021 / deepfake 相关关键词）。

    若主题明确为 LA/DF 降 EER，优先使用 search_asvspoof2021_eer_research；
    若关注图像域迁移，优先使用 search_deepfake_cross_domain。
    """
    q = query.strip()
    queries = [
        f"{q} ASVspoof 2021 LA DF EER 2024 2025 anti-spoofing",
        f"{q} audio deepfake detection arxiv github 2024 2025 2026",
        f"{q} ASVspoof 5 speech spoofing countermeasure",
    ]
    return _web_search(queries)


@tool
def list_evaluation_metrics() -> str:
    """列出 ASVspoof 2021 LA/DF 核心评测指标（EER）与实验规范。"""
    return f"""## 核心优化目标
- **首要指标**：{PRIMARY_METRIC}（Equal Error Rate），在 **ASVspoof 2021 LA** 与 **ASVspoof 2021 DF** 上均力求降低
- **辅助指标**：min t-DCF（官方代价）；论文常同时报告 LA 与 DF
- **说明**：本工具仅描述协议，**不提供当前 SOTA 的 EER 数值**；具体榜单请用检索工具查 2023–2026 文献

## ASVspoof 2021 LA / DF
- LA：逻辑访问伪造；DF：深度伪造音频；须**分别**报告 EER
- 可关注 ASVspoof 5 等新评测趋势，但主任务仍以 2021 LA/DF 为准时需明确协议

## 报告规范
- 分别给出 LA、DF 的 EER；固定随机种子；说明增强是否仅用训练集
- 对照基线须为**近年可复现**方法，并注明文献/仓库来源（勿默认写 AASIST 除非检索确认）

## 跨域借鉴（图像 deepfake / face anti-spoofing）
- 迁移时需在 LA、DF 上**分别验证** EER 是否下降，避免只对 LA 过拟合"""


@tool
def read_loaded_paper(paper_filename: str, section_hint: str = "") -> str:
    """读取已通过命令行加载的本地论文 PDF 全文或相关段落。

    Args:
        paper_filename: PDF 文件名，如 ``AASIST.pdf``
        section_hint: 可选关键词，如 ``EER``、``ASVspoof 2021``、``method``、``deepfake``
    """
    return read_paper_from_store(paper_filename, section_hint)


RESEARCH_TOOLS = [
    query_antispoofing_knowledge,
    search_asvspoof2021_eer_research,
    search_deepfake_cross_domain,
    search_open_research,
    list_evaluation_metrics,
    read_loaded_paper,
]

# Phase 工具注册表（Pipeline Gate 可按阶段筛选子集）
PHASE_TOOL_REGISTRY: dict[str, tuple] = {
    "query_antispoofing_knowledge": (query_antispoofing_knowledge, ToolCategory.PHASE),
    "search_asvspoof2021_eer_research": (search_asvspoof2021_eer_research, ToolCategory.PHASE),
    "search_deepfake_cross_domain": (search_deepfake_cross_domain, ToolCategory.PHASE),
    "search_open_research": (search_open_research, ToolCategory.PHASE),
    "list_evaluation_metrics": (list_evaluation_metrics, ToolCategory.PHASE),
    "read_loaded_paper": (read_loaded_paper, ToolCategory.PHASE),
}


def get_research_tools(phases: list[str] | None = None):
    """返回研究工具列表；phases 预留按阶段过滤（Pipeline Gate 轻量实现）。"""
    _ = phases
    return list(RESEARCH_TOOLS)


def get_reviewer_tools():
    """检查 Agent 与调研子 Agent 共用 Phase 工具（检索、读 PDF、协议说明）。"""
    return get_research_tools()


def collect_pdf_paths(
    pdf_args: list[str] | None = None,
    papers_dir: str | Path | None = None,
) -> list[Path]:
    """合并 --pdf 显式路径与 papers 目录下的所有 PDF。"""
    paths: list[Path] = []
    seen: set[str] = set()

    for raw in pdf_args or []:
        p = Path(raw).expanduser().resolve()
        key = str(p)
        if key not in seen and p.suffix.lower() == ".pdf":
            paths.append(p)
            seen.add(key)

    if papers_dir:
        folder = Path(papers_dir).expanduser().resolve()
        if folder.is_dir():
            for p in sorted(folder.glob("*.pdf")):
                key = str(p.resolve())
                if key not in seen:
                    paths.append(p)
                    seen.add(key)

    return paths
