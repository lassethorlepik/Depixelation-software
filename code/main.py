import os
import subprocess
import sys

actions = {
    "1": ("Synthesize images", "synthesizer.data_synth"),
    "2": ("Train model", "trainer"),
    "3": ("Benchmark model", "benchmarker"),
    "4": ("Run inference", "inference"),
}

BASE_DIR = os.path.dirname(__file__)


def run_action(module: str, description: str) -> None:
    """Execute a Python module as a subprocess."""
    print(f"\n>> {description}\n")
    subprocess.check_call([
        sys.executable,
        "-m",
        module,
    ], cwd=BASE_DIR)


def main() -> None:
    print("=== Command-Line Interface ===")
    menu_lines = ["\nSelect an option:"]
    for key, (desc, _) in actions.items():
        menu_lines.append(f" [{key}] {desc}")
    menu_lines.append(" [0] Exit\n")
    menu_text = "\n".join(menu_lines)
    while True:
        choice = input(menu_text + "\nEnter choice (index): ").strip()
        if choice == "0":
            break
        action = actions.get(choice)
        if action:
            description, module = action
            run_action(module, description)
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
