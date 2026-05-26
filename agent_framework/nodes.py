"""图节点：LLM 推理与工具执行前的绑定逻辑。"""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable

from agent_framework.config import SYSTEM_PROMPT


def create_agent_node(llm_with_tools: Runnable):
    """创建 agent 节点：调用绑定了工具的 LLM。"""

    def agent_node(state: dict) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent_node
