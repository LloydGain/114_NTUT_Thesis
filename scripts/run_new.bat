@echo off
REM ================================
REM Parallel run for multiple file_date
REM ================================

set PYTHON=python
set SCRIPT=..\src\main.py

set SEED=0
set DATES=20230113

for %%D in (%DATES%) do (
    echo Starting file_date=%%D
    start "run_%%D" %PYTHON% %SCRIPT% --file_date %%D --seed %SEED%
)

echo All jobs started.
pause