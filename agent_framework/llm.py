"""LLM 工厂：统一创建 ChatOpenAI 实例。"""

import socket
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_framework.config import (
    AgentRole,
    LLMConfig,
    get_llm_config,
    get_llm_max_retries,
    get_llm_timeout,
)


def create_llm(
    config: LLMConfig | None = None,
    *,
    role: AgentRole | None = None,
) -> ChatOpenAI:
    """
    创建 LLM 实例。

    Args:
        config: 显式配置；若省略则按 role 从环境变量加载（含分角色 temperature）
        role: orchestrator | sub_agent | reviewer；chat 模式可不传
    """
    cfg = config or get_llm_config(role=role)
    return ChatOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        model=cfg.model,
        temperature=cfg.temperature,
        timeout=get_llm_timeout(),
        max_retries=get_llm_max_retries(),
    )


def check_llm_connection(role: AgentRole | None = "orchestrator") -> str:
    """
    启动前检测：DNS 解析 + 一次最小 API 调用。
    失败时抛出带排查提示的 RuntimeError。
    """
    cfg = get_llm_config(role=role)
    host = urlparse(cfg.base_url).hostname
    if not host:
        raise RuntimeError(f"无效的 API 地址：{cfg.base_url}")

    try:
        socket.getaddrinfo(host, 443)
    except socket.gaierror as e:
        raise RuntimeError(
            f"无法解析 API 域名「{host}」（DNS 失败：{e}）。\n"
            "请检查：1) 本机网络/代理/VPN；2) .env 中 DEEPSEEK_BASE_URL 是否为 https://api.deepseek.com；\n"
            "3) 若在校园网/公司网，可能需要配置 HTTP_PROXY/HTTPS_PROXY。"
        ) from e

    llm = create_llm(config=cfg, role=role)
    try:
        resp = llm.invoke([HumanMessage(content="回复：ok")])
        return (resp.content or "")[:80]
    except Exception as e:
        raise RuntimeError(
            f"已解析域名「{host}」，但 API 请求失败：{e}\n"
            "请检查 API Key 是否有效、余额/配额，以及 base_url 是否与服务商文档一致。"
        ) from e
