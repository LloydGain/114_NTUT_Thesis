@echo off
chcp 65001 > nul
setlocal

:: ---[ Settings ]---------------------------------------------

set FILE_DATES=20221208
set SEEDS=1
set GOOGLE_FLAG=--google
set SKIP_COMPARE_FLAG=--skip_compare
set ALB_FLAG=

:: ---[ Run ]--------------------------------------------------

cd /d "%~dp0..\..\src"

set ALB_ARG=
if defined ALB_FLAG set ALB_ARG=--alb %ALB_FLAG%

for %%D in (%FILE_DATES%) do (
    echo.
    echo ============================================================
    echo  Running dataset: %%D
    echo ============================================================
    python "..\tools\run_experiments.py" --file_date %%D --seeds %SEEDS% %GOOGLE_FLAG% %SKIP_COMPARE_FLAG% %ALB_ARG%
    if errorlevel 1 (
        echo [ERROR] Dataset %%D failed. Continuing to next...
    )
)

echo.
echo ============================================================
echo  All datasets finished.
echo ============================================================
pause
