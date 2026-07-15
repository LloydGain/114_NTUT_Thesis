@echo off
echo Installing Python dependencies...
pip install -r "%~dp0..\..\requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo Python dependencies installed successfully!
pause
