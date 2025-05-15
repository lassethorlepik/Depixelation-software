@echo off
set VENV_PATH=venv

if not exist "%VENV_PATH%\Scripts\activate" (
    echo Creating virtual environment...
    python -m venv %VENV_PATH%
    echo Virtual environment created at %VENV_PATH%.
)

echo You can now modify venv modules from command line.
cmd /k "%VENV_PATH%\Scripts\activate"
