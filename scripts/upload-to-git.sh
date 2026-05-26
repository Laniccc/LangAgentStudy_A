#!/usr/bin/env bash
# 将当前项目代码提交并推送到 Git 远程仓库（Git Bash / WSL / macOS / Linux）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MESSAGE=""
REMOTE=""
BRANCH="main"
SKIP_PUSH=0

usage() {
  cat <<'EOF'
用法:
  ./scripts/upload-to-git.sh [-m "提交说明"] [-r "远程URL"] [-b 分支] [--skip-push]

示例:
  ./scripts/upload-to-git.sh -m "初始化项目"
  ./scripts/upload-to-git.sh -r "https://github.com/user/repo.git" -m "首次提交"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message) MESSAGE="$2"; shift 2 ;;
    -r|--remote)  REMOTE="$2"; shift 2 ;;
    -b|--branch)  BRANCH="$2"; shift 2 ;;
    --skip-push)  SKIP_PUSH=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "未知参数: $1"; usage; exit 1 ;;
  esac
done

step() { printf '\n>> %s\n' "$1"; }
fail() { printf '错误: %s\n' "$1" >&2; exit 1; }

command -v git >/dev/null 2>&1 || fail "未找到 git，请先安装 Git"

[[ -n "$MESSAGE" ]] || MESSAGE="更新代码 $(date '+%Y-%m-%d %H:%M')"

step "项目目录: $PROJECT_ROOT"

if [[ ! -d .git ]]; then
  step "初始化本地仓库 (git init)"
  git init -b "$BRANCH"
fi

if ! git remote | grep -q .; then
  if [[ -z "$REMOTE" ]]; then
    fail "尚未配置远程仓库。示例: ./scripts/upload-to-git.sh -r https://github.com/user/repo.git -m 首次提交"
  fi
  step "添加远程 origin: $REMOTE"
  git remote add origin "$REMOTE"
elif [[ -n "$REMOTE" ]]; then
  step "更新远程 origin: $REMOTE"
  git remote set-url origin "$REMOTE"
fi

[[ -f .env ]] && echo "提示: .env 已由 .gitignore 排除，不会上传密钥。"

step "暂存变更 (git add -A)"
git add -A

if [[ -z "$(git status --porcelain)" ]]; then
  echo "没有需要提交的变更。"
  if [[ "$SKIP_PUSH" -eq 0 ]]; then
    step "尝试推送到远程 ($BRANCH)"
    git push -u origin "$BRANCH" || true
  fi
  exit 0
fi

step "提交 (git commit)"
git commit -m "$MESSAGE" || fail "git commit 失败，请配置 user.name / user.email"

if [[ "$SKIP_PUSH" -eq 1 ]]; then
  echo "已本地提交，未推送（--skip-push）。"
  exit 0
fi

step "推送到 origin/$BRANCH"
git push -u origin "$BRANCH" || fail "git push 失败，请检查远程 URL 与权限"

echo ""
echo "完成: 代码已提交并推送到 origin/$BRANCH"
git log -1 --oneline
