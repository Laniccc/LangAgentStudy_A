"""语音鉴伪研究专用工具。"""

import json
from pathlib import Path

from langchain_core.tools import tool

from agent_framework.research.pdf_loader import read_paper_from_store

# 领域知识库（开源可达成果摘要，供检索失败时回退）
_ANTISPOOFING_KB = {
    "benchmarks": {
        "ASVspoof 2019 LA": "逻辑访问场景；指标 EER、min t-DCF；基线 LFCC-GMM、LCNN 等",
        "ASVspoof 2021 DF": "深度伪造；强调跨域与未知攻击",
        "ASVspoof 5": "多场景统一评测趋势",
        "ADD 2022/2023": "音频深度伪造检测挑战赛",
    },
    "models": {
        "AASIST": "频谱图 + 异构图注意力；ASVspoof 2021 LA 强基线",
        "RawNet2/3": "端到端 Raw waveform CNN",
        "WavLM/SSL-front": "自监督前端 + 后端分类器",
        "LFCC-LCNN": "经典手工特征基线",
        "FAD": "融合辅助任务、多尺度建模",
    },
    "techniques": {
        "frontend": "Raw waveform、LEAF、SincNet、Wav2Vec/WavLM 特征",
        "pooling": "Attentive statistics pooling、ASTP",
        "augmentation": "RawBoost、MUSAN noise、codec 模拟、混响",
        "loss": "A-Softmax、AM-Softmax、二分类 CE + 对比学习",
        "generalization": "DG、MMD、对抗训练、多域混合",
    },
}


@tool
def query_antispoofing_knowledge(topic: str) -> str:
    """查询语音鉴伪领域知识：benchmarks / models / techniques 或综合摘要。

    topic 示例：benchmarks、AASIST、数据增强、域泛化
    """
    topic_lower = topic.lower().strip()
    parts = []

    for category, items in _ANTISPOOFING_KB.items():
        if topic_lower in category or category in topic_lower:
            parts.append(f"## {category}\n{json.dumps(items, ensure_ascii=False, indent=2)}")

    for category, items in _ANTISPOOFING_KB.items():
        for key, value in items.items():
            if topic_lower in key.lower() or topic_lower in value.lower():
                parts.append(f"- **{key}** ({category}): {value}")

    if not parts:
        return json.dumps(_ANTISPOOFING_KB, ensure_ascii=False, indent=2)
    return "\n".join(parts)


@tool
def search_open_research(query: str) -> str:
    """搜索互联网上语音鉴伪/深度伪造检测相关的开源研究与资源。

    用于查找论文标题、GitHub 仓库、技术博客等。query 建议使用英文关键词。
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=6))
        if not hits:
            return f"未检索到结果。建议换用关键词：anti-spoofing {query} github arxiv"
        lines = []
        for i, h in enumerate(hits, 1):
            title = h.get("title", "")
            body = h.get("body", "")[:300]
            href = h.get("href", "")
            lines.append(f"{i}. **{title}**\n   {body}\n   {href}")
        return "\n\n".join(lines)
    except ImportError:
        return (
            f"（未安装 duckduckgo-search，无法联网检索）\n"
            f"请基于关键词手动调研：{query}\n"
            f"可参考：arxiv anti-spoofing、ASVspoof challenge github、AASIST pytorch"
        )
    except Exception as e:
        return f"检索异常：{e}。请换关键词重试或使用 query_antispoofing_knowledge。"


@tool
def list_evaluation_metrics() -> str:
    """列出语音鉴伪常用评测指标与注意事项。"""
    return """## 常用指标
- **EER** (Equal Error Rate)：越低越好；需报告开发集调参、评估集测试结果
- **min t-DCF**：ASVspoof 官方代价函数；注意 CM 系统与 ASV 联合场景
- **Accuracy / AUC**：辅助参考，不能替代 EER

## 实验注意
- 固定随机种子与训练/验证划分
- 报告参数量、推理 RTF、显存
- 跨域测试（训练 LA、测试 DF 等）验证泛化
- 避免验证集泄漏到增强策略选择中"""


@tool
def read_loaded_paper(paper_filename: str, section_hint: str = "") -> str:
    """读取已通过命令行加载的本地论文 PDF 全文或相关段落。

    Args:
        paper_filename: PDF 文件名，如 ``AASIST.pdf``
        section_hint: 可选关键词，用于定位摘要/方法/实验等段落，如 ``method``、``EER``、``attention``
    """
    return read_paper_from_store(paper_filename, section_hint)


RESEARCH_TOOLS = [
    query_antispoofing_knowledge,
    search_open_research,
    list_evaluation_metrics,
    read_loaded_paper,
]


def get_research_tools():
    return list(RESEARCH_TOOLS)


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
