@echo off
rem Double-click to start the VolleyIQ web app. It opens in your browser.
rem Keep this window open while you use the app; close it (or press Ctrl+C) to stop.
title VolleyIQ
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Could not find the Python environment in "%~dp0.venv".
    echo See "Setup" in README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m volleyiq.webapp %*
if errorlevel 1 pause
