"""环境变量与运行配置。"""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0


def get_llm_config() -> LLMConfig:
    """优先使用 DeepSeek，否则回退到 OpenAI 兼容配置。"""
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        return LLMConfig(
            api_key=deepseek_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError(
            "未找到 API Key。请在 .env 中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
        )

    return LLMConfig(
        api_key=openai_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )

SYSTEM_PROMPT = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    "你是一个有帮助的 AI 助手。在需要时使用工具完成任务，回答要简洁准确。",
)

