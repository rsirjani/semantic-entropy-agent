"""Docker container management for SWE-bench agent interaction."""

import logging
import os
import subprocess
import tempfile
import docker
import time

logger = logging.getLogger(__name__)


def clone_container_state(
    source_container_id: str,
    target_container_id: str,
    workdir: str = "/testbed",
) -> None:
    """Clone modified files from source container to target container via docker cp.

    Steps:
    1. Get list of modified/untracked files from source via git
    2. Copy each modified file from source to target
    """
    # Get modified files (tracked changes + untracked files)
    result = subprocess.run(
        ["docker", "exec", "-w", workdir, source_container_id,
         "bash", "-c",
         "git diff --name-only HEAD 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        logger.warning(f"Failed to get modified files: {result.stderr}")
        return

    modified_files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
    if not modified_files:
        logger.info("No modified files to clone")
        return

    logger.info(f"Cloning {len(modified_files)} modified files to new container")

    with tempfile.TemporaryDirectory() as tmpdir:
        for filepath in modified_files:
            src_path = f"{workdir}/{filepath}"
            local_path = os.path.join(tmpdir, filepath)

            # Create local directory structure
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Copy from source container to local
            try:
                subprocess.run(
                    ["docker", "cp", f"{source_container_id}:{src_path}", local_path],
                    capture_output=True, text=True, timeout=30, check=True,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to copy {filepath} from source: {e.stderr}")
                continue

            # Ensure target directory exists
            target_dir = os.path.dirname(src_path)
            subprocess.run(
                ["docker", "exec", target_container_id, "mkdir", "-p", target_dir],
                capture_output=True, timeout=10,
            )

            # Copy from local to target container
            try:
                subprocess.run(
                    ["docker", "cp", local_path, f"{target_container_id}:{src_path}"],
                    capture_output=True, text=True, timeout=30, check=True,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to copy {filepath} to target: {e.stderr}")

    logger.info(f"Cloned {len(modified_files)} files successfully")


class SWEBenchContainer:
    """Manages a Docker container for agent bash execution against a SWE-bench instance."""

    def __init__(self, instance_id: str, image_name: str | None = None):
        self.instance_id = instance_id
        self.image_name = image_name or f"sweb.eval.x86_64.{instance_id}:latest"
        self.client = docker.from_env()
        self.container = None

    def start(self) -> None:
        """Start the container in detached mode."""
        self.container = self.client.containers.run(
            image=self.image_name,
            detach=True,
            tty=True,
            stdin_open=True,
            working_dir="/testbed",
            entrypoint="/bin/bash",
            command=["-c", "sleep infinity"],
        )
        # Wait for container to be ready
        time.sleep(2)
        self.container.reload()
        if self.container.status != "running":
            raise RuntimeError(
                f"Container failed to start. Status: {self.container.status}"
            )

        # Reset any prior changes so we start clean
        self.exec_bash("cd /testbed && git checkout -- . && git clean -fd", timeout=30)

    def exec_bash(self, command: str, timeout: int = 120) -> tuple[str, int]:
        """Execute a bash command in the container.

        Returns (output, exit_code).
        """
        if self.container is None:
            raise RuntimeError("Container not started. Call start() first.")

        try:
            exec_result = self.container.exec_run(
                cmd=["bash", "-c", command],
                workdir="/testbed",
                user="root",
                demux=True,
            )

            # demux=True returns (stdout, stderr) tuple
            stdout = exec_result.output[0] or b""
            stderr = exec_result.output[1] or b""

            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                err_text = stderr.decode("utf-8", errors="replace")
                if err_text.strip():
                    output = output + "\nSTDERR:\n" + err_text

            return output, exec_result.exit_code

        except Exception as e:
            return f"Error executing command: {e}", 1

    def get_patch(self) -> str:
        """Get git diff from the container (the agent's changes)."""
        output, exit_code = self.exec_bash(
            "cd /testbed && git diff --no-color"
        )
        if exit_code != 0:
            return ""
        return output.strip()

    def cleanup(self) -> None:
        """Stop and remove the container."""
        if self.container:
            try:
                self.container.stop(timeout=10)
            except Exception:
                pass
            try:
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
