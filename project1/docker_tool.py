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

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_BASE_DIR, "output")
_CODE_DIR = os.path.join(_BASE_DIR, ".code_tmp")
_CONTAINER_ID_FILE = os.path.join(_BASE_DIR, ".container_id")

def _get_or_create_container() -> "docker.models.containers.Container":
    client = docker.from_env()
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    os.makedirs(_CODE_DIR, exist_ok=True)

    if os.path.exists(_CONTAINER_ID_FILE):
        try:
            with open(_CONTAINER_ID_FILE) as f:
                container = client.containers.get(f.read().strip())
            if container.status == "running":
                return container
            container.remove()
        except Exception:
            pass

    container = client.containers.run(
        "code-executor:latest",
        "sleep infinity",
        volumes={
            _CODE_DIR: {"bind": "/app", "mode": "rw"},
            _OUTPUT_DIR: {"bind": "/output", "mode": "rw"},
        },
        detach=True,
        mem_limit="256m",   # max RAM the container can use
        cpu_quota=50000,    # 50000 = 50% of 1 CPU core (out of 100000 = 1 full core) 
    )
    with open(_CONTAINER_ID_FILE, "w") as f:
        f.write(container.id)
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
    container = _get_or_create_container()
    code_file = f"/app/code_{uuid.uuid4().hex[:8]}.py"
    host_file = os.path.join(_CODE_DIR, os.path.basename(code_file))

    with open(host_file, "w") as f:
        f.write(code.strip())

    t0 = time.time()
    try:
        if packages:
            container.exec_run(f"uv pip install --system -q {' '.join(packages)}", stdout=True, stderr=True)

        exit_code, output = container.exec_run(
            f"python {code_file}", stdout=True, stderr=True, workdir="/app"
        )
        result = output.decode("utf-8")
        if exit_code != 0 and len(result) > 500:
            result = result[:500] + "\n...[truncated]"
        saved_files = [
            os.path.join(_OUTPUT_DIR, f) for f in os.listdir(_OUTPUT_DIR)
            if os.path.getmtime(os.path.join(_OUTPUT_DIR, f)) >= t0
        ]
        if saved_files:
            result += f"\nSaved files: {saved_files}"
        return result
    finally:
        if os.path.exists(host_file):
            os.unlink(host_file)

if __name__ == "__main__":
    print(execute_docker.invoke({
        "code": """
import cowsay
cowsay.cow("Hi Kshitij")
""",
        "packages": ["cowsay"]
    }))
