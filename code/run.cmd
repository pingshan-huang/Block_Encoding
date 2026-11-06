@echo off
setlocal
set "PROJECT_PYTHON=%~dp0..\.venv\Scripts\python.exe"

if exist "%PROJECT_PYTHON%" (
    "%PROJECT_PYTHON%" "%~dp0main.py" %*
    exit /b %errorlevel%
)

py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0main.py" %*
    exit /b %errorlevel%
)

python -c "import sys" >nul 2>nul
if not errorlevel 1 (
    python "%~dp0main.py" %*
    exit /b %errorlevel%
)

echo Python was not found.
echo Install Python 3 and run:
echo   py -3 -m pip install -r "%~dp0requirements.txt"
exit /b 1
