import os
import subprocess
import sys
import platform

def create_venv(venv_path):
    """Create a virtual environment if it doesn't exist."""
    if not os.path.exists(venv_path):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', venv_path])
        print("Virtual environment created.")

def install_requirements(venv_path):
    """Install packages from requirements.txt into the virtual environment."""
    print("Installing dependencies...")
    # os.path.join for cross-platform compatibility
    pip_path = os.path.join(venv_path, 'bin' if os.name == 'posix' else 'Scripts', 'pip')
    subprocess.check_call([pip_path, 'install', '-r', 'requirements.txt'])
    print("Dependencies installed.")

if __name__ == "__main__":
    print(f"Python version: {platform.python_version()}")
    # Change the current working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    venv_path = "venv"
    create_venv(venv_path)
    install_requirements(venv_path)
    input("Press Enter to exit...")
