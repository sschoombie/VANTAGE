@echo off
setlocal

REM ===== Required versions =====
set REQUIRED_PYTHON=3.12.2
set VENV_NAME=venv_vantage

echo Checking Python installation...

REM ---- Find python executables in PATH ----
set PY_FOUND=
for /f "delims=" %%p in ('where python') do (
    for /f "tokens=2" %%v in ('"%%p" --version 2^>^&1') do (
        echo Checking Python at %%p: version %%v
        if "%%v"=="%REQUIRED_PYTHON%" (
            set PY_FOUND=%%p
            goto :PYTHON_OK
        )
    )
)

:PYTHON_OK
if "%PY_FOUND%"=="" (
    echo.
    echo No Python %REQUIRED_PYTHON% installation found in PATH.
    echo Please install Python %REQUIRED_PYTHON% from:
    echo https://www.python.org/downloads/release/python-3122/
    echo IMPORTANT: Check "Add Python to PATH" during install.
    pause
    exit /b
)

echo Using Python: %PY_FOUND%

REM ===== Check ffmpeg =====
ffmpeg -version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo FFmpeg is not installed or not in PATH.
    echo Please install FFmpeg from:
    echo https://ffmpeg.org/download.html
    echo After installation, make sure ffmpeg.exe is in your system PATH.
    pause
    exit /b
)

echo FFmpeg is installed.

REM ===== Create virtual environment =====
IF NOT EXIST %VENV_NAME% (
    echo Creating virtual environment...
    "%PY_FOUND%" -m venv %VENV_NAME%
) ELSE (
    echo Virtual environment already exists.
)

REM ===== Activate venv and install dependencies =====
call %VENV_NAME%\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

IF EXIST requirements.txt (
    echo Installing dependencies...
    pip install -r requirements.txt
) ELSE (
    echo requirements.txt not found.
    pause
    exit /b
)

deactivate

echo Setup complete! You can now run the script using run.bat
pause