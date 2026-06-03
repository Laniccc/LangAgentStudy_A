# 代码上传到 Git — 使用说明

本项目提供一键脚本：自动 `git add` → `commit` → `push`。密钥文件（`.env`）已通过 `.gitignore` 排除。

---

## 一、首次使用前准备（只需做一次）

### 1. 安装 Git

- Windows: https://git-scm.com/download/win  
- 安装后打开 **PowerShell** 或 **Git Bash**，执行 `git --version` 确认可用。

### 2. 配置身份（全局，一次即可）

```powershell
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

### 3. 在 GitHub / Gitee 创建空仓库

- 在网页上 **New repository**，**不要**勾选「用 README 初始化」（避免与本地首次推送冲突）。
- 复制仓库地址，例如：
  - HTTPS: `https://github.com/你的用户名/LangAgent_A.git`
  - SSH: `git@github.com:你的用户名/LangAgent_A.git`

  https://github.com/Laniccc/LangAgentStudy_A.git

### 4. 认证方式（二选一）

| 方式 | 说明 |
|------|------|
| **HTTPS + Token** | GitHub: Settings → Developer settings → Personal access token；推送时密码处填 Token。 |
| **SSH** | 生成密钥 `ssh-keygen -t ed25519`，公钥添加到 GitHub/Gitee；远程 URL 用 `git@...` 形式。 |

---

## 二、日常上传命令

在项目根目录 `LangAgent_A` 下执行。

### PowerShell（推荐，Windows）

```powershell
# 进入项目根目录
cd "e:\学习文件\研究生\就业\Agent学习\LangAgent_A"

# 首次：指定远程地址 + 提交说明
.\scripts\upload-to-git.ps1 -Remote "https://github.com/你的用户名/LangAgent_A.git" -Message "首次提交：LangAgent_A"

# 之后每次改完代码
.\scripts\upload-to-git.ps1 -Message "加入子agent返回内容上下文压缩（github仓库借鉴Headroom）"

# 只提交到本地，暂不推送
.\scripts\upload-to-git.ps1 -Message "本地备份" -SkipPush
```

### Git Bash / WSL / macOS

```bash
cd "/e/学习文件/研究生/就业/Agent学习/LangAgent_A"   # 路径按实际修改

chmod +x scripts/upload-to-git.sh   # 首次赋予执行权限

./scripts/upload-to-git.sh -r "https://github.com/你的用户名/LangAgent_A.git" -m "首次提交"
./scripts/upload-to-git.sh -m "日常更新说明"
```

### CMD 或双击

```cmd
cd /d "e:\学习文件\研究生\就业\Agent学习\LangAgent_A"
scripts\upload-to-git.bat -Message "更新说明"
```

带远程地址（首次）：

```cmd
scripts\upload-to-git.bat -Remote "https://github.com/你的用户名/LangAgent_A.git" -Message "首次提交"
```

---

## 三、脚本参数说明

| 参数 (PowerShell) | 参数 (Shell) | 说明 |
|-------------------|--------------|------|
| `-Message` | `-m` / `--message` | 本次提交说明 |
| `-Remote` | `-r` / `--remote` | 远程仓库 URL；无 `origin` 时自动添加 |
| `-Branch` | `-b` / `--branch` | 分支名，默认 `main` |
| `-SkipPush` | `--skip-push` | 只 commit，不 push |

未写 `-Message` 时，会自动生成：`更新代码 2026-05-26 14:30` 这类说明。

---

## 四、脚本实际执行的 Git 步骤

1. 若尚无 `.git` → `git init -b main`
2. 若无 `origin` 且提供了 `-Remote` → `git remote add origin <URL>`
3. `git add -A`（遵守 `.gitignore`）
4. `git commit -m "<你的说明>"`
5. `git push -u origin main`

---

## 五、常见问题

**Q: 提示 `fatal: not a git repository`**  
A: 在项目根目录运行脚本，不要进到 `scripts` 子目录单独执行（`.bat` 会自动切到根目录）。

**Q: `git push` 被拒绝（rejected）**  
A: 远程已有 README 等提交。可二选一：  
- 在 GitHub 上删掉仓库重建为**完全空仓库**后再推送；或  
- `git pull origin main --rebase` 解决冲突后再 `.\scripts\upload-to-git.ps1`。

**Q: 不想上传某些文件**  
A: 编辑项目根目录 `.gitignore`，把路径或通配符加进去后重新运行脚本。

**Q: `.env` 会被上传吗？**  
A: 不会，已在 `.gitignore` 中。请用 `.env.example` 作为模板，密钥只留在本机。

---

## 六、不用脚本时的等价手动命令

```powershell
cd "e:\学习文件\研究生\就业\Agent学习\LangAgent_A"
git init -b main
git remote add origin https://github.com/你的用户名/LangAgent_A.git
git add -A
git commit -m "首次提交"
git push -u origin main
```

日常更新：

```powershell
git add -A
git commit -m "你的说明"
git push
```
