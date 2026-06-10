"""Docker container management for SWE-bench agent interaction."""

import logging
import os
import subprocess
import tempfile
import docker
import time

logger = logging.getLogger(__name__)


class ContainerCloneError(RuntimeError):
    """A fork's filesystem clone could not be completed faithfully.

    Raised instead of silently degrading: fork-state consistency is
    load-bearing for the SDLG arm (the fork's container must start from
    exactly the parent's working tree, or the child trajectory's results are
    attributed to the SDLG mechanism while actually running from a different
    state). The caller (`PhasedOrchestrator._clone_for_sdlg`) catches any
    exception and registers the draw as failed-at-creation (R7.2), which is
    the honest accounting for an unclonable fork.
    """


def clone_container_state(
    source_container_id: str,
    target_container_id: str,
    workdir: str = "/testbed",
) -> None:
    """Replicate the source container's working-tree state onto the target.

    Strict, deletion-aware contract (raises ContainerCloneError on ANY step
    it cannot complete — never a partial/silent clone):

    1. List tracked changes with status letters (`git diff --name-status -z`,
       rename detection disabled so renames appear as D + A) and untracked
       files (`git ls-files --others --exclude-standard -z`).
    2. DELETED files are deleted in the target (docker cp cannot copy a file
       that no longer exists — the old implementation warned and left the
       file alive in the fork, silently desyncing its state).
    3. Modified/added/untracked files are copied via docker cp, with every
       copy and mkdir checked.

    NUL-separated listings (`-z`) keep filenames with spaces intact. Note the
    common case is a no-op: SDLG forks at the FIRST detected write, before it
    executes, so the parent tree is typically pristine; the strict contract
    matters for the VERIFY-phase SDLG fallback and undetectable writes
    (`python -c "open(...,'w')"`), where the parent tree can be dirty.
    """
    def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise ContainerCloneError(
                f"{' '.join(cmd[:4])}... failed (rc={result.returncode}): "
                f"{(result.stderr or result.stdout or '').strip()[:500]}"
            )
        return result

    tracked = _run([
        "docker", "exec", "-w", workdir, source_container_id,
        "git", "-c", "diff.renames=false", "diff", "--name-status", "-z", "HEAD",
    ])
    untracked = _run([
        "docker", "exec", "-w", workdir, source_container_id,
        "git", "ls-files", "--others", "--exclude-standard", "-z",
    ])

    deleted: list[str] = []
    to_copy: list[str] = []
    tokens = [t for t in tracked.stdout.split("\0") if t]
    for status, path in zip(tokens[::2], tokens[1::2]):
        if status.startswith("D"):
            deleted.append(path)
        else:  # M / A / T (renames disabled -> no R/C entries)
            to_copy.append(path)
    to_copy.extend(t for t in untracked.stdout.split("\0") if t)

    if not deleted and not to_copy:
        logger.info("No modified files to clone")
        return

    logger.info(
        f"Cloning container state: {len(to_copy)} modified/untracked file(s), "
        f"{len(deleted)} deletion(s)"
    )

    for filepath in deleted:
        _run(["docker", "exec", "-w", workdir, target_container_id,
              "rm", "-f", f"{workdir}/{filepath}"], timeout=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, filepath in enumerate(to_copy):
            src_path = f"{workdir}/{filepath}"
            # Flat local name (index prefix) sidesteps Windows path-separator
            # and collision issues; the container path keeps the real layout.
            local_path = os.path.join(tmpdir, f"{i}_{os.path.basename(filepath)}")
            _run(["docker", "cp", f"{source_container_id}:{src_path}", local_path])
            target_dir = os.path.dirname(src_path)
            if target_dir:
                _run(["docker", "exec", target_container_id,
                      "mkdir", "-p", target_dir], timeout=10)
            _run(["docker", "cp", local_path, f"{target_container_id}:{src_path}"])

    logger.info(f"Cloned {len(to_copy)} file(s) (+{len(deleted)} deletion(s)) successfully")


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
