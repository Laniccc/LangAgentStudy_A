"""LangGraph ReAct Agent 图构建。"""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent_framework.llm import create_llm
from agent_framework.nodes import create_agent_node
from agent_framework.state import AgentState
from agent_framework.tools import get_tools


def build_agent_graph(tools=None, checkpointer=None):
    """
    构建 ReAct 循环图：

        START -> agent -> (有 tool_calls?) -> tools -> agent -> ... -> END
    """
    tool_list = tools if tools is not None else get_tools()
    llm = create_llm()
    llm_with_tools = llm.bind_tools(tool_list)

    graph = StateGraph(AgentState)
    graph.add_node("agent", create_agent_node(llm_with_tools))
    graph.add_node("tools", ToolNode(tool_list))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)


def create_agent(tools=None, checkpointer=None):
    """编译并返回可执行的 Agent 应用。"""
    return build_agent_graph(tools=tools, checkpointer=checkpointer)
