@echo off
REM CleanPass - builds the plugin for every Nuke 16 / 17 installed.
REM Close Nuke first. Extra arguments are passed to build.ps1, e.g.:
REM   build.bat -Nuke "C:\Program Files\Nuke17.0v3"
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" %*
echo.
pause
