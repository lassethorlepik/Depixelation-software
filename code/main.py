import os
import subprocess
import sys


def synthesize():
    print("\n>> Synthesize images\n")
    subprocess.check_call(
        [sys.executable, "-m", "synthesizer.data_synth"],
        cwd=os.path.dirname(__file__)
    )


def train_model():
    print("\n>> Train model\n")
    subprocess.check_call(
        [sys.executable, "-m", "trainer"],
        cwd=os.path.dirname(__file__)
    )


def benchmark_model():
    print("\n>> Benchmark model\n")
    subprocess.check_call(
        [sys.executable, "-m", "benchmarker"],
        cwd=os.path.dirname(__file__)
    )


def run_inference():
    print("\n>> Run inference\n")
    subprocess.check_call(
        [sys.executable, "-m", "inference"],
        cwd=os.path.dirname(__file__)
    )


def main():
    print("=== Command-Line Interface ===")

    menu = (
        "\nSelect an option:\n"
        " [1] Synthesize images\n"
        " [2] Train model\n"
        " [3] Benchmark model\n"
        " [4] Run inference\n"
        " [0] Exit\n"
    )

    while True:
        choice = input(menu + "Enter choice (index): ").strip()
        if choice == "1":
            synthesize()
        elif choice == "2":
            train_model()
        elif choice == "3":
            benchmark_model()
        elif choice == "4":
            run_inference()
        elif choice == "0":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
