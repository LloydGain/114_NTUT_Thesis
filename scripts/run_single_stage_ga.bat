@echo off
chcp 65001 > nul
setlocal

:: ---[ Settings ]---------------------------------------------

set FILE_DATES=20221203
set SEEDS=1
set TEST_MODE=0
set FORCE_MODE=1
set GOOGLE_MODE=0

if "%TEST_MODE%"=="1" (
    set TEST_FLAG=--test
) else (
    set TEST_FLAG=
)

if "%FORCE_MODE%"=="1" (
    set FORCE_FLAG=--force
) else (
    set FORCE_FLAG=
)

if "%GOOGLE_MODE%"=="1" (
    set GOOGLE_FLAG=--google
) else (
    set GOOGLE_FLAG=
)

:: ---[ Run ]--------------------------------------------------

cd /d "%~dp0"

for %%D in (%FILE_DATES%) do (
    echo.
    echo ============================================================
    if "%TEST_MODE%"=="1" (
        echo  Running dataset: %%D - Single-Stage GA [TEST MODE]
    ) else (
        echo  Running dataset: %%D - Single-Stage GA
    )
    echo ============================================================
    python "..\tools\run_single_stage_ga.py" --file_date %%D --seeds %SEEDS% %TEST_FLAG% %FORCE_FLAG% %GOOGLE_FLAG%
    if errorlevel 1 (
        echo [ERROR] Dataset %%D failed. Continuing to next...
    )
)

echo.
echo ============================================================
echo  All datasets finished.
echo ============================================================
pause
