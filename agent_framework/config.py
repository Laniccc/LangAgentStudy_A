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
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
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
