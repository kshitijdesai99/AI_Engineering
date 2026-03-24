"""
LangChain tool that runs arbitrary Python code inside a local Docker container.
Provides isolation, memory/CPU limits, and cleanup. Used by docker_agent.py.
"""
import docker
from langchain_core.tools import tool
import tempfile
import os

@tool
def execute_docker(code: str, packages: list[str] = None, timeout: int = 30) -> str:
    """
    Execute Python code in a Docker container.
    
    Args:
        code: Python code to execute
        packages: Optional list of pip packages to install before running (e.g. ["seaborn", "pandas"])
        timeout: Timeout in seconds
        
    Returns:
        Output from code execution and paths to any saved files
    """
    client = docker.from_env()
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Create temporary file with code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    print(f"\n--- Executing in Docker ---\n{code}\n---------------------------\n")
    try:
        # Run code in container
        result = client.containers.run(
            "code-executor:latest",
            f"sh -c 'pip install -q {' '.join(packages)} && python {os.path.basename(temp_file)}'" if packages else f"python {os.path.basename(temp_file)}",
            volumes={
                os.path.dirname(temp_file): {'bind': '/app', 'mode': 'ro'},
                output_dir: {'bind': '/output', 'mode': 'rw'},
            },
            working_dir='/app',
            remove=True,
            stdout=True,
            stderr=True,
            mem_limit='128m',
            cpu_quota=50000
        )

        output = result.decode('utf-8')
        saved_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir)]
        if saved_files:
            output += f"\nSaved files: {saved_files}"
        return output

    except docker.errors.ContainerError as e:
        return f"Error: {e.stderr.decode('utf-8')}"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

if __name__ == "__main__":
    print(execute_docker.invoke("print(3+2-1)"))
