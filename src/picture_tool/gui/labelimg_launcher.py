"""LabelImg launcher and integration module."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class LabelImgLauncher:
    """Handles LabelImg launching and configuration."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.labelimg_executable = self._find_labelimg()

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
        except Exception as e:
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
        except Exception as e:
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
        if not self.is_installed():
            logger.error("LabelImg is not installed")
            return False
        
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return False
        
        try:
            # Build command
            if str(self.labelimg_executable).endswith(".py"):
                # Run vendored script
                cmd = [sys.executable, self.labelimg_executable]
            elif self.labelimg_executable == "python_module":
                cmd = [sys.executable, "-m", "labelImg"]
            else:
                cmd = [self.labelimg_executable]
            
            # Add arguments
            cmd.append(str(input_dir))
            
            if predefined_classes_file and predefined_classes_file.exists():
                cmd.append(str(predefined_classes_file))
            
            cmd.append(str(output_dir))
            
            logger.info(f"Launching LabelImg with command: {' '.join(cmd)}")
            
            # Launch as subprocess
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(self.labelimg_executable).parent) if str(self.labelimg_executable).endswith(".py") else None
            )
            
            logger.info(f"LabelImg launched with PID: {self.process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch LabelImg: {e}")
            return False

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
