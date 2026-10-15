@echo off
setlocal
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PROJECT_PYTHON%" (
    echo Project Python environment not found.
    echo Create it and install dependencies with:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)

"%PROJECT_PYTHON%" "%~dp0main.py" %*
