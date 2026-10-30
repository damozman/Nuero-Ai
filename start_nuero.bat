@echo off
rem Starts the app from anywhere: double-click this file, or run .\start_nuero in a terminal.
rem   .\start_nuero local   - use the local AI (Ollama) for this session, starting Ollama if needed
cd /d "%~dp0nuero_ai_memory"

if /i "%~1"=="local" goto local

if defined GEMINI_API_KEY goto run
if defined ANTHROPIC_API_KEY goto run
if defined XAI_API_KEY goto run

echo No AI key is set. Which AI service should it use?
echo   1) Gemini (Google)
echo   2) Claude (Anthropic)
echo   3) Grok (xAI)
echo   4) Local AI on this computer (Ollama) - private, works offline
choice /c 1234 /n /m "Press 1, 2, 3 or 4: "
if errorlevel 4 goto local
if errorlevel 3 (set "KEYVAR=XAI_API_KEY") else if errorlevel 2 (set "KEYVAR=ANTHROPIC_API_KEY") else (set "KEYVAR=GEMINI_API_KEY")
set /p KEYVALUE=Paste your API key and press Enter:
set "%KEYVAR%=%KEYVALUE%"
set "KEYVALUE="
goto run

:local
set "NUERO_PROVIDER=ollama"
curl -s -m 2 http://127.0.0.1:11434/api/version >nul 2>&1 && goto run
set "OLLAMA_EXE=ollama"
where ollama >nul 2>&1 || set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not exist "%OLLAMA_EXE%" if /i not "%OLLAMA_EXE%"=="ollama" (
    echo Ollama isn't installed. Get it from https://ollama.com, then run this again.
    pause
    exit /b 1
)
echo Starting Ollama...
start "Ollama" /min "%OLLAMA_EXE%" serve
for /l %%i in (1,1,20) do (
    curl -s -m 2 http://127.0.0.1:11434/api/version >nul 2>&1 && goto run
    timeout /t 1 /nobreak >nul
)
echo Ollama didn't start in time. Open the Ollama app, then pick it in the dashboard's AI menu.

:run
".venv\Scripts\python.exe" main.py
pause
