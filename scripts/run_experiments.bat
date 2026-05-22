@echo off
chcp 65001 > nul
setlocal

:: ---[ Settings ]---------------------------------------------

set FILE_DATES=20260109 20260106 20260102 20230113 20230107 20230102 20221231 20221230 20221227 20221223 20221220 20221213 20221212 20221212 20221210 20221203
set SEEDS=5
set GOOGLE_FLAG=--google
set SKIP_COMPARE_FLAG=--skip_compare

:: ---[ Run ]--------------------------------------------------

cd /d "%~dp0"

for %%D in (%FILE_DATES%) do (
    echo.
    echo ============================================================
    echo  Running dataset: %%D
    echo ============================================================
    python "..\tools\run_experiments.py" --file_date %%D --seeds %SEEDS% %GOOGLE_FLAG% %SKIP_COMPARE_FLAG%
    if errorlevel 1 (
        echo [ERROR] Dataset %%D failed. Continuing to next...
    )
)

echo.
echo ============================================================
echo  All datasets finished.
echo ============================================================
pause
