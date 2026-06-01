# 在 conda 环境 claudcode 中安装/更新项目依赖
# 用法（PowerShell）: .\scripts\setup_conda.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$envName = "claudcode"

Write-Host "==> 检查 conda ..."
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "未找到 conda，请先安装 Anaconda/Miniconda 并加入 PATH。"
}

$exists = conda env list | Select-String "^\s*$envName\s"
if (-not $exists) {
    Write-Host "==> 创建环境 $envName (Python 3.11) ..."
    conda env create -f environment.yml
} else {
    Write-Host "==> 更新环境 $envName ..."
    conda env update -f environment.yml --prune
}

Write-Host "==> 修复二进制 wheel（pydantic / jiter，conda 升级 Python 后可能需要）..."
conda run -n $envName pip install --force-reinstall pydantic pydantic-core -q
conda run -n $envName pip install --force-reinstall --no-cache-dir jiter -q

Write-Host "==> 验证安装 ..."
conda run -n $envName python -c @"
from agent_framework import create_agent, create_research_agent
from agent_framework.research.tools import _get_ddgs_class, get_research_tools
print('create_agent OK')
print('create_research_agent OK')
print('research tools:', [t.name for t in get_research_tools()])
print('DDGS:', _get_ddgs_class())
"@

Write-Host ""
Write-Host "完成。激活环境："
Write-Host "  conda activate $envName"
Write-Host "运行研究流程："
Write-Host "  python main.py --mode research"
