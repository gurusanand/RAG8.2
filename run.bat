@echo off
echo ============================================================
echo   7-Layer Advanced RAG - Banking Assistant
echo   Starting Application...
echo ============================================================
echo.

:: Check virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Load .env file if it exists
if exist ".env" (
    echo Loading environment variables from .env...
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
)

:: Check API key
if "%OPENAI_API_KEY%"=="" (
    echo [WARNING] OPENAI_API_KEY is not set!
    echo Please set it in the .env file or as an environment variable.
    echo.
)

echo.
echo Starting Streamlit server...
echo The app will open in your default browser at http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.

streamlit run app.py --server.port 8501 --server.address localhost

pause
