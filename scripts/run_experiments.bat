@echo off
chcp 65001 > nul
setlocal

:: ---[ Settings ]---------------------------------------------

set FILE_DATES=20221203 20221205 20221207 20221208 20221209 20221210 20221212 20221213 20221214 20221215 20221216 20221217 20221219 20221220 20221221 20221222 20221223 20221227 20221230 20221231 20230102 20230107 20230113 20260102 20260106 20260109
set SEEDS=10
set GOOGLE_FLAG=--google
set SKIP_COMPARE_FLAG=--skip_compare
set ALB_FLAG=--alb extract

:: ---[ Run ]--------------------------------------------------

cd /d "%~dp0"

for %%D in (%FILE_DATES%) do (
    echo.
    echo ============================================================
    echo  Running dataset: %%D
    echo ============================================================
    python "..\tools\run_experiments.py" --file_date %%D --seeds %SEEDS% %GOOGLE_FLAG% %SKIP_COMPARE_FLAG% %ALB_FLAG%
    if errorlevel 1 (
        echo [ERROR] Dataset %%D failed. Continuing to next...
    )
)

echo.
echo ============================================================
echo  All datasets finished.
echo ============================================================
pause
