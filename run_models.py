import subprocess
import time
from pathlib import Path

LOG_FILE = Path("execution_times.log")

def run_and_time(script_path: str):
    start_time = time.perf_counter()

    # Run the Python script
    subprocess.run(
        ["python", script_path],
        check=True
    )

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # Log the result
    with LOG_FILE.open("a+") as log:
        log.write(f"{script_path} executed in {elapsed:.4f} seconds\n")

    return elapsed

if __name__ == "__main__":
    # run_and_time('voxtral_test_data.py')
    run_and_time('flamingo_test_data.py')
    run_and_time('qwen_test_data.py')
