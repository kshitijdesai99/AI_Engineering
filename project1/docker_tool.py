"""
LangChain tool that runs arbitrary Python code inside a persistent Docker container.
The container is started once and reused across script runs via a stored container ID.
Provides isolation, memory/CPU limits, and cleanup. Used by docker_agent.py.
"""
import docker
from langchain_core.tools import tool
import os
import time
import uuid

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # Directory of this script
_OUTPUT_DIR = os.path.join(_BASE_DIR, "output")              # Where saved files (plots, etc.) are written
_CODE_DIR = os.path.join(_BASE_DIR, ".code_tmp")             # Temp dir mounted into the container as /app
_CONTAINER_ID_FILE = os.path.join(_BASE_DIR, ".container_id") # Persists container ID across runs

def _get_or_create_container() -> "docker.models.containers.Container":
    """Reuse an existing running container or spin up a new one."""
    client = docker.from_env()
    os.makedirs(_OUTPUT_DIR, exist_ok=True)  # Ensure output dir exists
    os.makedirs(_CODE_DIR, exist_ok=True)    # Ensure code temp dir exists

    # ── reuse existing container ─────────────────────────────
    if os.path.exists(_CONTAINER_ID_FILE):
        try:
            with open(_CONTAINER_ID_FILE) as f:
                container = client.containers.get(f.read().strip())  # Look up container by saved ID
            if container.status == "running":
                return container  # Reuse if still running
            container.remove()    # Clean up stopped container
        except Exception:
            pass  # Container not found or invalid — fall through to create a new one

    # ── create new container ─────────────────────────────────
    container = client.containers.run(
        "code-executor:latest",
        "sleep infinity",      # Keep container alive indefinitely
        volumes={
            _CODE_DIR: {"bind": "/app", "mode": "rw"},       # Mount code dir
            _OUTPUT_DIR: {"bind": "/output", "mode": "rw"},  # Mount output dir
        },
        detach=True,
        mem_limit="256m",   # max RAM the container can use
        cpu_quota=50000,    # 50000 = 50% of 1 CPU core (out of 100000 = 1 full core)
    )
    with open(_CONTAINER_ID_FILE, "w") as f:
        f.write(container.id)  # Persist container ID for reuse on next run
    return container

@tool
def execute_docker(code: str, packages: list[str] = None, timeout: int = 60) -> str:
    """
    Execute Python code in a persistent Docker container.

    Args:
        code: Python code to execute
        packages: Optional list of pip packages to install (e.g. ["seaborn", "pandas"])
        timeout: Timeout in seconds

    Returns:
        Output from code execution and paths to any saved files
    """
    # ── setup ────────────────────────────────────────────────
    container = _get_or_create_container()
    code_file = f"/app/code_{uuid.uuid4().hex[:8]}.py"           # Unique filename inside container
    host_file = os.path.join(_CODE_DIR, os.path.basename(code_file))  # Corresponding path on host

    with open(host_file, "w") as f:
        f.write(code.replace("\\n", "\n").strip())  # Write code to temp file (normalize newlines)

    t0 = time.time()  # Record start time to detect newly saved files after execution
    try:
        # ── install packages ─────────────────────────────────
        if packages:
            container.exec_run(f"uv pip install --system -q {' '.join(packages)}", stdout=True, stderr=True) # Install packages in the container

        # ── execute code ─────────────────────────────────────
        exit_code, output = container.exec_run(
            f"python {code_file}", stdout=True, stderr=True, workdir="/app"
        )
        result = output.decode("utf-8")
        if exit_code != 0 and len(result) > 500:
            result = result[:500] + "\n...[truncated]"  # Truncate long error output

        # ── collect saved files ──────────────────────────────
        saved_files = [
            os.path.join(_OUTPUT_DIR, f) for f in os.listdir(_OUTPUT_DIR)
            if os.path.getmtime(os.path.join(_OUTPUT_DIR, f)) >= t0  # Only files created during this run
        ]
        if saved_files:
            result += f"\nSaved files: {saved_files}"
        return result
    finally:
        if os.path.exists(host_file):
            os.unlink(host_file)  # Always clean up temp code file

if __name__ == "__main__":
    # ── smoke test ───────────────────────────────────────────
    print(execute_docker.invoke({
        "code": """
import cowsay
cowsay.cow("Hi Kshitij")
""",
        "packages": ["cowsay"]
    }))
