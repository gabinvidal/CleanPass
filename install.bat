@echo off
cd /d "%~dp0"
python install.py
if errorlevel 1 py install.py
pause
