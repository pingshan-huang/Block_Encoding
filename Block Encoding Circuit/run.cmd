@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PROJECT_ROOT=%~dp0.."
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    "%PROJECT_ROOT%\.venv\Scripts\python.exe" "%~dp0main.py" %*
) else (
    py -3 "%~dp0main.py" %*
)
