"""检测 LLM API 的 DNS 与连通性。用法：python scripts/check_api_connection.py"""

import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

# 保证可导入 agent_framework
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from agent_framework.config import get_llm_config
    from agent_framework.llm import check_llm_connection

    cfg = get_llm_config()
    host = urlparse(cfg.base_url).hostname or "(无效 URL)"
    print(f"base_url : {cfg.base_url}")
    print(f"model    : {cfg.model}")
    print(f"api_key  : {'已设置 (' + cfg.api_key[:8] + '...)' if cfg.api_key else '未设置'}")

    print(f"\n[1/2] DNS 解析 {host} ...")
    try:
        addrs = socket.getaddrinfo(host, 443)
        print(f"  成功，示例 IP：{addrs[0][4][0]}")
    except socket.gaierror as e:
        print(f"  失败：{e}")
        print("\n建议：检查网络/代理；.env 中 DEEPSEEK_BASE_URL=https://api.deepseek.com")
        return 1

    print("\n[2/2] API 试调用 ...")
    try:
        text = check_llm_connection()
        print(f"  成功，回复：{text}")
    except RuntimeError as e:
        print(f"  失败：{e}")
        return 1

    print("\n全部通过，可运行 python main.py --mode research")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
