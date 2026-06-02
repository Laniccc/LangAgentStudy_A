#Requires -Version 5.1
<#
.SYNOPSIS
    将当前项目代码提交并推送到 Git 远程仓库。

.PARAMETER Message
    提交说明；默认带时间戳。

.PARAMETER Remote
    远程仓库 URL。若尚未配置 origin，会用此地址执行 git remote add。

.PARAMETER Branch
    分支名，默认 main。

.PARAMETER SkipPush
    仅本地 add + commit，不 push。

.EXAMPLE
    .\scripts\upload-to-git.ps1 -Message "初始化 LangAgent_A"

.EXAMPLE
    .\scripts\upload-to-git.ps1 -Remote "https://github.com/你的用户名/LangAgent_A.git"
#>
param(
    [string]$Message = "",
    [string]$Remote = "",
    [string]$Branch = "main",
    [switch]$SkipPush
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Write-Step($text) { Write-Host "`n>> $text" -ForegroundColor Cyan }
function Fail($text) {
    Write-Host "错误: $text" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "未找到 git，请先安装 Git: https://git-scm.com/download/win"
}

if ([string]::IsNullOrWhiteSpace($Message)) {
    $Message = "更新代码 $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
}

Write-Step "项目目录: $ProjectRoot"

if (-not (Test-Path ".git")) {
    Write-Step "初始化本地仓库 (git init)"
    git init -b $Branch
    if ($LASTEXITCODE -ne 0) { Fail "git init 失败" }
}

$remotes = @(git remote 2>$null)
if ($remotes.Count -eq 0) {
    if ([string]::IsNullOrWhiteSpace($Remote)) {
        Fail @"
尚未配置远程仓库。请先创建 GitHub/Gitee 空仓库，然后任选其一：

  1) 带参数运行（推荐首次）:
     .\scripts\upload-to-git.ps1 -Remote "https://github.com/你的用户名/仓库名.git" -Message "首次提交"

  2) 手动配置后重跑:
     git remote add origin https://github.com/你的用户名/仓库名.git
     .\scripts\upload-to-git.ps1 -Message "首次提交"
"@
    }
    Write-Step "添加远程 origin: $Remote"
    git remote add origin $Remote
    if ($LASTEXITCODE -ne 0) { Fail "git remote add 失败" }
} elseif (-not [string]::IsNullOrWhiteSpace($Remote)) {
    Write-Step "更新远程 origin: $Remote"
    git remote set-url origin $Remote
    if ($LASTEXITCODE -ne 0) { Fail "git remote set-url 失败" }
}

if (Test-Path ".env") {
    Write-Host "提示: 检测到 .env，已由 .gitignore 排除，不会上传密钥。" -ForegroundColor Yellow
}

Write-Step "暂存变更 (git add -A)"
git add -A
if ($LASTEXITCODE -ne 0) { Fail "git add 失败" }

$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "没有需要提交的变更。" -ForegroundColor Green
    if (-not $SkipPush) {
        Write-Step "尝试推送到远程 ($Branch)"
        git push -u origin $Branch 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "推送完成（无新提交）。" -ForegroundColor Green
        }
    }
    exit 0
}

Write-Step "提交 (git commit)"
git commit -m $Message
if ($LASTEXITCODE -ne 0) {
    Fail @"
git commit 失败。若提示未配置用户信息，请先执行（仅需一次）:

  git config --global user.name "你的名字"
  git config --global user.email "你的邮箱"
"@
}

if ($SkipPush) {
    Write-Host "`n已本地提交，未推送（-SkipPush）。" -ForegroundColor Green
    exit 0
}

Write-Step "推送到 origin/$Branch"
git push -u origin $Branch
if ($LASTEXITCODE -ne 0) {
    Fail @"
git push 失败。常见原因:
  - 远程仓库不存在或无权限（检查 URL、SSH 密钥或 Personal Access Token）
  - 远程已有提交且历史不一致：先在网页创建空仓库，或执行 git pull origin $Branch --rebase 后再推送
"@
}

Write-Host "`n完成: 代码已提交并推送到 origin/$Branch" -ForegroundColor Green
git log -1 --oneline
