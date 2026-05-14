@echo off
chcp 65001 > nul
setlocal

:: ---[ Settings ]---------------------------------------------

set FILE_DATES=20221203 20221205 20221207
set SEEDS=5
set GOOGLE_FLAG=--google

:: ---[ Run ]--------------------------------------------------

cd /d "%~dp0"

for %%D in (%FILE_DATES%) do (
    echo.
    echo ============================================================
    echo  Running dataset: %%D
    echo ============================================================
    python "..\tools\run_experiments.py" --file_date %%D --seeds %SEEDS% %GOOGLE_FLAG%
    if errorlevel 1 (
        echo [ERROR] Dataset %%D failed. Continuing to next...
    )
)

echo.
echo ============================================================
echo  All datasets finished.
echo ============================================================
pause
