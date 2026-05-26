"""LLM 工厂：统一创建 ChatOpenAI 实例。"""

from langchain_openai import ChatOpenAI

from agent_framework.config import LLMConfig, get_llm_config


def create_llm(config: LLMConfig | None = None) -> ChatOpenAI:
    cfg = config or get_llm_config()
    return ChatOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=cfg.temperature,
    )
