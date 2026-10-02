@echo off
setlocal
cd /d "%~dp0"
echo ========================================
echo   AI Health Guardian - Windows Launcher
echo ========================================
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Could not create Python 3.11 environment.
    echo Install Python 3.11 and run this file again.
    pause
    exit /b 1
  )
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app\app.py
endlocal
