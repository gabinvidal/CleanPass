@echo off
REM CleanPass - installer. Close Nuke first.
REM Uses the Python that ships with Nuke (any installed version), else a system Python.
cd /d "%~dp0"
set "PY="
for /d %%D in ("%ProgramFiles%\Nuke*") do if exist "%%~D\python.exe" set "PY=%%~D\python.exe"
if defined PY (
    "%PY%" install.py
) else (
    py -3 install.py 2>nul || python install.py
)
pause
