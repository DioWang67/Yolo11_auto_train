import logging
import subprocess
import shutil
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class DVCWrapper:
    """Helper to interact with DVC via subprocess."""

    def __init__(self, cwd: Path = Path(".")):
        self.cwd = cwd
        self._dvc_cmd = shutil.which("dvc")

    @property
    def is_installed(self) -> bool:
        return self._dvc_cmd is not None

    @property
    def is_dvc_repo(self) -> bool:
        return (self.cwd / ".dvc").is_dir()

    def run_cmd(self, args: List[str]) -> bool:
        if not self.is_installed:
            logger.warning("DVC is not installed or not in PATH.")
            return False

        cmd = ["dvc"] + args
        try:
            logger.info(f"Running DVC command: {' '.join(cmd)}")
            subprocess.run(
                cmd,
                cwd=self.cwd,
                check=True,
                capture_output=False,  # Let output flow to stdout for user visibility
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC command failed: {e}")
            return False
        except Exception as e:
            logger.error(f"dvc execution error: {e}")
            return False

    def init(self) -> bool:
        return self.run_cmd(["init"])

    def pull(self) -> bool:
        return self.run_cmd(["pull"])

    def status(self) -> bool:
        return self.run_cmd(["status"])

    def add(self, targets: List[str]) -> bool:
        return self.run_cmd(["add"] + targets)
