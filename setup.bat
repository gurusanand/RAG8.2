@echo off
echo ============================================================
echo   7-Layer Advanced RAG - Banking Assistant
echo   Windows Local Setup Script
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [1/5] Python found.
python --version
echo.

:: Create virtual environment
echo [2/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo     Virtual environment created.
) else (
    echo     Virtual environment already exists.
)
echo.

:: Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo     Virtual environment activated.
echo.

:: Install dependencies
echo [4/5] Installing dependencies (this may take a few minutes)...
pip install --upgrade pip
pip install -r requirements.txt
echo.
echo     Dependencies installed successfully.
echo.

:: Check .env file
echo [5/5] Checking configuration...
if not exist ".env" (
    echo [WARNING] .env file not found!
    echo Please create a .env file with your OpenAI API key.
    echo.
    echo     Copy .env.example to .env and fill in your API key:
    echo       copy .env.example .env
    echo       notepad .env
    echo.
) else (
    echo     .env file found.
)

:: Create data directories
if not exist "data\cache" mkdir data\cache
if not exist "data\feedback" mkdir data\feedback
if not exist "data\faiss_index" mkdir data\faiss_index
echo     Data directories ready.
echo.

echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo   To start the application, run:
echo     run.bat
echo.
echo   Or manually:
echo     venv\Scripts\activate
echo     streamlit run app.py
echo.
pause
