@echo off
REM ================================
REM Sequential run for multiple file_date
REM ================================

set PYTHON=python
set SCRIPT=..\src\main.py

set SEED=0
set DATA_DIR=..\data
set "COMMENT=With Local Search (OSRM * 1.75)"

echo Scanning directories in %DATA_DIR%...

for /d %%D in ("%DATA_DIR%\*") do (
    call :process_folder "%%~nxD"
)

echo All jobs finished.
pause
goto :EOF


:process_folder
set "FOLDER=%~1"

REM Check if the folder name is numeric
echo %FOLDER%| findstr "^[0-9][0-9]*$" >nul
if errorlevel 1 (
    echo Skipping non-numeric folder: %FOLDER%
    goto :EOF
)

echo Running file_date=%FOLDER%

%PYTHON% %SCRIPT% --file_date %FOLDER% --seed %SEED% --comment "%COMMENT%"

echo Finished file_date=%FOLDER%
echo --------------------------------

goto :EOF