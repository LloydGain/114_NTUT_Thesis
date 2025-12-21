@echo off
REM ================================
REM Parallel run for multiple file_date
REM ================================

set PYTHON=python
set SCRIPT=main.py

set SEED=0
set DATES=1203 1205 1207 1208 1209 1210 1212 1213 1214 1215

for %%D in (%DATES%) do (
    echo Starting file_date=%%D
    start "run_%%D" %PYTHON% %SCRIPT% --file_date %%D --seed %SEED% --google
)

echo All jobs started.
pause