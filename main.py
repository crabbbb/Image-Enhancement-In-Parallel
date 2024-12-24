import os
import sys
import subprocess
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import json
import webbrowser
import socket
import time
import asyncio

# Fix for zmq warning on Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def add_environment_to_jupyter():
    """
    Dynamically adds the current Python environment to Jupyter as a kernel.
    If the kernel already exists, it skips the installation.
    """
    env_name = os.path.basename(sys.prefix)
    display_name = f"Python ({env_name})"
    kernel_name = env_name.replace(" ", "_").lower()

    print(f"Checking if environment '{env_name}' is already added as a Jupyter kernel...")

    # Check existing kernels
    try:
        existing_kernels = subprocess.check_output(
            [sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
            universal_newlines=True
        )
        kernels = json.loads(existing_kernels).get('kernelspecs', {})

        if kernel_name in kernels:
            print(f"Kernel '{kernel_name}' already exists. Skipping installation.")
            return kernel_name
    except Exception as e:
        print(f"Could not verify existing kernels: {e}")

    print(f"Adding environment '{env_name}' as a Jupyter kernel...")

    # Run the ipykernel installation command
    subprocess.run(
        [
            sys.executable, "-m", "ipykernel", "install",
            "--user",
            f"--name={kernel_name}",
            f"--display-name={display_name}"
        ],
        check=True
    )

    return kernel_name


def execute_python_file(file_path):
    """
    Executes a given Python file.
    """
    print(f"Executing Python file: {file_path}")
    subprocess.run([sys.executable, file_path], check=True)


def run_ipynb_file(ipynb_path, kernel_name):
    """
    Opens and runs all cells in a Jupyter notebook using the specified kernel.
    """
    print(f"Running IPython Notebook: {ipynb_path} with kernel '{kernel_name}'")

    # Read the notebook
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
        
    # -----------------------------------
    # Force the notebook metadata to reference your desired kernel
    # (so that when you open the notebook, it will use this kernel).
    # -----------------------------------
    notebook.metadata["kernelspec"] = {
        "name": kernel_name,
        "display_name": f"Python ({kernel_name})",
        "language": "python",
    }

    # Execute the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name=kernel_name)
    ep.preprocess(notebook, {'metadata': {'path': os.getcwd()}})

    # Save the new output notebook on the notebook
    with open(ipynb_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    print(f"Executed notebook, writing output to notebook.")

def is_port_in_use(port):
    """
    Checks if a given port is in use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) == 0


def find_free_port(start_port=8888):
    """
    Finds the next available port starting from `start_port`.
    """
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port

def wait_for_jupyter(port, timeout=15):
    """
    Wait up to `timeout` seconds for Jupyter to start listening on localhost:port.
    Returns True if the server is detected, False if the timeout expires.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False

def open_notebook_in_browser(ipynb_path, kernel_name):
    """
    Launch a Jupyter notebook server without auto-opening a browser
    and wait until the port is ready before printing any messages.
    """
    print("Checking for available port...")
    port = find_free_port()

    print(f"Launching Jupyter Notebook server for '{ipynb_path}' on port {port}...")

    # Start Jupyter WITHOUT automatically opening a browser
    process = subprocess.Popen(
        [
            sys.executable,
            "-m", "jupyter", "notebook", ipynb_path,
            f"--port={port}",
            "--no-browser",  # Ensure Jupyter doesn't open a browser by itself
            "--NotebookApp.shutdown_no_activity_timeout=10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # (Python 3.7+) for universal_newlines
    )

    # Instead of reading the logs, we just poll the port until it's open
    ready = wait_for_jupyter(port, timeout=15)
    
    if ready:
        # Once the server is up, print your message
        print(f"Notebook '{ipynb_path}' is now running on port {port}.")
        # If you want to open a browser tab at that point:
        import webbrowser
        url = f"http://localhost:{port}/notebooks/{os.path.basename(ipynb_path)}"
        webbrowser.open(url)
    else:
        print("Jupyter server did not start within 15 seconds. Check logs for any errors.")
        # (Optional) If desired, you could terminate the process here:
        # process.terminate()

    return process

if __name__ == "__main__":
    # Paths to the known Python file and IPython notebook
    python_file = "testing/testing.py"
    ipynb_file = "testing/testing.ipynb"

    # Add the current environment to Jupyter as a kernel
    kernel_name = add_environment_to_jupyter()

    # Execute the Python file
    execute_python_file(python_file)

    # Run the IPython notebook with the dynamically added kernel
    run_ipynb_file(ipynb_file, kernel_name)

    # Open the Jupyter notebook in a browser
    notebook_process = open_notebook_in_browser(ipynb_file, kernel_name)
