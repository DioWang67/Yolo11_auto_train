"""LabelImg launcher and integration module."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, List, Optional

logger = logging.getLogger(__name__)


class LabelImgLauncher:
    """Handles LabelImg launching and configuration."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.labelimg_executable = self._find_labelimg()
        self.last_error: Optional[str] = None
        self.last_log_path: Optional[Path] = None

    def _find_labelimg(self) -> Optional[str]:
        """Find LabelImg executable in the system."""
        # 1. Try vendored version (Priority)
        # Assuming we are in src/picture_tool/gui/labelimg_launcher.py
        # Vendor path: src/picture_tool/libs/labelImg/labelImg.py
        try:
            current_dir = Path(__file__).parent
            vendor_path = current_dir.parent / "libs" / "labelImg" / "labelImg.py"
            if vendor_path.exists():
                logger.info(f"Found vendored LabelImg at: {vendor_path}")
                return str(vendor_path)
        except (OSError, AttributeError) as e:
            logger.warning(f"Error checking vendor path: {e}")

        # 2. Try to find labelImg command
        executable = shutil.which("labelImg")
        if executable:
            logger.info(f"Found labelImg at: {executable}")
            return executable

        # 3. Try Python module execution
        try:
            result = subprocess.run(
                [sys.executable, "-m", "labelImg", "--help"],
                capture_output=True,
                timeout=2,
            )
            if result.returncode == 0 or b"usage" in result.stdout.lower():
                logger.info("labelImg can be executed as Python module")
                return "python_module"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        logger.warning("labelImg not found in system PATH")
        return None

    def is_installed(self) -> bool:
        """Check if LabelImg is installed."""
        return self.labelimg_executable is not None

    def prepare_environment(
        self,
        classes: List[str],
        input_dir: Path,
        output_dir: Path,
    ) -> bool:
        """Prepare environment for LabelImg.

        Args:
            classes: List of class names
            input_dir: Directory containing images to label
            output_dir: Directory for output labels

        Returns:
            True if preparation successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create predefined_classes.txt for LabelImg
            classes_file = output_dir.parent / "predefined_classes.txt"
            with open(classes_file, "w", encoding="utf-8") as f:
                for class_name in classes:
                    f.write(f"{class_name}\n")

            logger.info(f"Created predefined_classes.txt at {classes_file}")
            logger.info(f"Classes: {classes}")

            return True
        except (OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to prepare environment: {e}")
            return False

    def launch(
        self,
        input_dir: Path,
        output_dir: Path,
        predefined_classes_file: Optional[Path] = None,
    ) -> bool:
        """Launch LabelImg with configured settings.

        Args:
            input_dir: Directory containing images
            output_dir: Directory for saving labels
            predefined_classes_file: Path to predefined_classes.txt

        Returns:
            True if launched successfully, False otherwise
        """
        self.last_error = None
        self.last_log_path = None

        if not self.is_installed():
            self.last_error = "LabelImg is not installed"
            logger.error(self.last_error)
            return False

        if not input_dir.exists():
            self.last_error = f"Input directory does not exist: {input_dir}"
            logger.error(self.last_error)
            return False

        try:
            # Build command
            if str(self.labelimg_executable).endswith(".py"):
                # Run vendored script
                cmd: List[str] = [sys.executable, str(self.labelimg_executable)]
            elif self.labelimg_executable == "python_module":
                cmd = [sys.executable, "-m", "labelImg"]
            else:
                cmd = [str(self.labelimg_executable)]

            # Add arguments
            cmd.append(str(input_dir))

            if predefined_classes_file and predefined_classes_file.exists():
                cmd.append(str(predefined_classes_file))

            cmd.append(str(output_dir))

            logger.info(f"Launching LabelImg with command: {' '.join(cmd)}")

            log_path = output_dir.parent / "labelimg_launch.log"
            stdout_log, stderr_log = self._open_launch_logs(log_path)
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            self.process = subprocess.Popen(
                cmd,
                stdout=stdout_log,
                stderr=stderr_log,
                cwd=str(Path(str(self.labelimg_executable)).parent)
                if str(self.labelimg_executable).endswith(".py")
                else None,
                creationflags=creationflags,
            )
            stdout_log.close()
            stderr_log.close()
            self.last_log_path = log_path

            logger.info(f"LabelImg launched with PID: {self.process.pid}")
            time.sleep(1.5)
            exit_code = self.process.poll()
            if exit_code is not None:
                detail = self._read_launch_log(log_path)
                self.last_error = (
                    f"LabelImg exited immediately with code {exit_code}."
                    f" See log: {log_path}"
                )
                if detail:
                    self.last_error = f"{self.last_error}\n{detail}"
                logger.error(self.last_error)
                return False

            return True

        except (OSError, RuntimeError, FileNotFoundError) as e:
            self.last_error = f"Failed to launch LabelImg: {e}"
            logger.error(self.last_error)
            return False

    def _open_launch_logs(self, log_path: Path) -> tuple[IO[str], IO[str]]:
        """Open stdout/stderr log handles for the LabelImg subprocess.

        Args:
            log_path: File path used for both stdout and stderr diagnostics.

        Returns:
            A pair of writable text file handles for stdout and stderr.

        Raises:
            OSError: If the parent directory cannot be created or the log cannot be opened.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_log = open(log_path, "w", encoding="utf-8")
        stderr_log = open(log_path, "a", encoding="utf-8")
        stdout_log.write("=== labelImg stdout ===\n")
        stdout_log.flush()
        stderr_log.write("\n=== labelImg stderr ===\n")
        stderr_log.flush()
        return stdout_log, stderr_log

    def _read_launch_log(self, log_path: Path, max_chars: int = 2000) -> str:
        """Read a bounded diagnostic tail from the LabelImg launch log.

        Args:
            log_path: Launch log path.
            max_chars: Maximum number of trailing characters to return.

        Returns:
            Bounded log text, or an empty string if unavailable.
        """
        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return content[-max_chars:].strip()

    def is_running(self) -> bool:
        """Check if LabelImg process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def wait_for_completion(self, timeout: Optional[float] = None) -> int:
        """Wait for LabelImg to close.

        Args:
            timeout: Maximum time to wait in seconds (None = infinite)

        Returns:
            Exit code of the process
        """
        if self.process is None:
            return -1

        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("LabelImg process wait timeout expired")
            return -1

    def terminate(self):
        """Terminate LabelImg process if running."""
        if self.process and self.is_running():
            logger.info("Terminating LabelImg process")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't terminate, killing it")
                self.process.kill()
