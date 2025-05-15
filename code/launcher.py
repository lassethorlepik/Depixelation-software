import os
import platform
import subprocess
import sys

if __name__ == "__main__":
    print(f"Python version: {platform.python_version()}")
    print(f"Current environment path: {sys.prefix}")
    python_executable = os.path.join("venv", 'bin' if os.name == 'posix' else 'Scripts', 'python.exe')
    script_path = os.path.join(os.path.abspath('.'), "main.py")
    subprocess.check_call([python_executable, script_path])
