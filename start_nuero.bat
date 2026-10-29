@echo off
rem Starts Nuero AI from anywhere: double-click this file, or run .\start_nuero in a terminal.
cd /d "%~dp0nuero_ai_memory"

if defined GEMINI_API_KEY goto run
if defined ANTHROPIC_API_KEY goto run
if defined XAI_API_KEY goto run

echo No AI key is set. Which AI service will Nuero AI use?
echo   1) Gemini (Google)
echo   2) Claude (Anthropic)
echo   3) Grok (xAI)
choice /c 123 /n /m "Press 1, 2 or 3: "
if errorlevel 3 (set "KEYVAR=XAI_API_KEY") else if errorlevel 2 (set "KEYVAR=ANTHROPIC_API_KEY") else (set "KEYVAR=GEMINI_API_KEY")
set /p KEYVALUE=Paste your API key and press Enter:
set "%KEYVAR%=%KEYVALUE%"
set "KEYVALUE="

:run
".venv\Scripts\python.exe" main.py
pause
