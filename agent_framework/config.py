"""环境变量与运行配置。"""

from dataclasses import dataclass
import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

AgentRole = Literal["orchestrator", "sub_agent", "reviewer"]

_ROLE_TEMPERATURE_ENV: dict[AgentRole, str] = {
    "orchestrator": "ORCHESTRATOR_TEMPERATURE",
    "sub_agent": "SUB_AGENT_TEMPERATURE",
    "reviewer": "REVIEWER_TEMPERATURE",
}

# 各角色默认 temperature（未在 .env 中设置时使用）
_ROLE_TEMPERATURE_DEFAULT: dict[AgentRole, float] = {
    "orchestrator": 0.2,
    "sub_agent": 0.1,
    "reviewer": 0.1,
}


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0


def _normalize_base_url(url: str) -> str:
    """去除空白并修正常见 URL 写法。"""
    url = (url or "").strip()
    if not url:
        return url
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url.rstrip("/")


def get_llm_timeout() -> float:
    return float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))


def get_llm_max_retries() -> int:
    return int(os.getenv("LLM_MAX_RETRIES", "3"))


def get_temperature_for_role(role: AgentRole | None = None) -> float:
    """按 Agent 角色读取 temperature；未配置的角色项回退到 LLM_TEMPERATURE。"""
    fallback = float(os.getenv("LLM_TEMPERATURE", "0"))
    if role is None:
        return fallback

    env_key = _ROLE_TEMPERATURE_ENV[role]
    raw = os.getenv(env_key)
    if raw is not None and raw.strip() != "":
        return float(raw)
    return _ROLE_TEMPERATURE_DEFAULT.get(role, fallback)


def get_llm_config(role: AgentRole | None = None) -> LLMConfig:
    """优先使用 DeepSeek，否则回退到 OpenAI 兼容配置。"""
    temperature = get_temperature_for_role(role)
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        return LLMConfig(
            api_key=deepseek_key.strip(),
            base_url=_normalize_base_url(
                os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            ),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro").strip(),
            temperature=temperature,
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError(
            "未找到 API Key。请在 .env 中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
        )

    return LLMConfig(
        api_key=openai_key.strip(),
        base_url=_normalize_base_url(
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        temperature=temperature,
    )


SYSTEM_PROMPT = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    "你是一个有帮助的 AI 助手。在需要时使用工具完成任务，回答要简洁准确。",
)

_LEGACY_DEEPSEEK_MODEL_HINTS: dict[str, str] = {
    "deepseek-chat": "旧名 → API 按 deepseek-v4-flash 计费；请改 DEEPSEEK_MODEL=deepseek-v4-pro 或 deepseek-v4-flash",
    "deepseek-reasoner": "旧名 → API 按 deepseek-v4-flash（思考模式）计费；请改 deepseek-v4-pro / deepseek-v4-flash",
}


def _mask_api_key(key: str) -> str:
    key = (key or "").strip()
    if not key:
        return "未设置"
    if len(key) <= 10:
        return "已设置"
    return f"已设置 ({key[:6]}…{key[-4:]})"


def format_llm_startup_banner(*, research_mode: bool = False) -> str:
    """生成启动时打印的 LLM 配置摘要（不含完整密钥）。"""
    cfg = get_llm_config()
    provider = "DeepSeek" if os.getenv("DEEPSEEK_API_KEY") else "OpenAI 兼容"
    lines = [
        "--- LLM 配置 ---",
        f"提供商 : {provider}",
        f"model  : {cfg.model}",
        f"base   : {cfg.base_url}",
        f"api_key: {_mask_api_key(cfg.api_key)}",
    ]
    legacy = _LEGACY_DEEPSEEK_MODEL_HINTS.get(cfg.model.lower())
    if legacy:
        lines.append(f"提示   : {legacy}")
    if research_mode:
        lines.append(
            "temperature: "
            f"主控 {get_temperature_for_role('orchestrator')} | "
            f"子Agent {get_temperature_for_role('sub_agent')} | "
            f"审阅 {get_temperature_for_role('reviewer')}"
        )
    else:
        lines.append(f"temperature: {get_temperature_for_role()}")
    lines.append("---")
    return "\n".join(lines)


def print_llm_startup_info(*, research_mode: bool = False) -> None:
    print(format_llm_startup_banner(research_mode=research_mode) + "\n")
