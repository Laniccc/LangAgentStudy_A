"""Agent 工具集：在此注册自定义工具。"""

from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """获取当前日期与时间（本地时区）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """计算简单数学表达式。仅支持数字与 + - * / ( ) 运算符。"""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含不允许的字符。"
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"计算失败：{e}"


DEFAULT_TOOLS = [get_current_time, calculator]


def get_tools():
    """返回当前 Agent 可用的工具列表。扩展时在此追加。"""
    return list(DEFAULT_TOOLS)
