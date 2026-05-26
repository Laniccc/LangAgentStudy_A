@echo off
REM 双击或在 cmd 中运行，调用 PowerShell 上传脚本
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0upload-to-git.ps1" %*
exit /b %ERRORLEVEL%
