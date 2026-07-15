@echo off
set PYTHON=python
set SCRIPT=..\..\tools\run_tune_hpo.py
set SEED=0

REM ===== file_date 清單 =====
set FILE_DATES=20221205 20221207 20221208 20221209 20221210

for %%F in (%FILE_DATES%) do (
    echo ================================
    echo Starting file_date=%%F at %time%

    %PYTHON% %SCRIPT% --file_date %%F --seed %SEED%

    if errorlevel 1 (
        echo ERROR on %%F !!!!
        pause
        exit /b
    )

    echo Finished file_date=%%F at %time%
)

echo All jobs done!
pause