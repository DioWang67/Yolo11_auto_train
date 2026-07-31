"""Picture Tool Application Entry Point.

This module initializes the QApplication and launches the MainWindow.
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Any
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication


# Test/integration injection point. Production keeps this ``None`` so the real
# window remains lazily imported after background resource limits are applied.
MainWindow: type[Any] | None = None


def _apply_background_resource_policy() -> None:
    """Keep unattended CPU training responsive to inspection and desktop work."""
    cpu_count = os.cpu_count() or 2
    default_threads = max(1, min(4, cpu_count // 2))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, str(default_threads))

    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        below_normal_priority_class = 0x00004000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if not kernel32.SetPriorityClass(process, below_normal_priority_class):
            error_code = ctypes.get_last_error()
            raise OSError(error_code, "SetPriorityClass failed")
    except (AttributeError, OSError) as exc:
        print(
            f"WARNING: unable to lower background training priority: {exc}",
            file=sys.stderr,
        )

def main() -> None:
    """Application Entry Point."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--handoff")
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--import-package")
    parser.add_argument("--background", action="store_true")
    args, qt_args = parser.parse_known_args()
    if args.background:
        _apply_background_resource_policy()

    # Import after applying thread limits so numerical libraries inherit them.
    window_class = MainWindow
    if window_class is None:
        from picture_tool.gui.main_window import MainWindow as RealMainWindow

        window_class = RealMainWindow
    if sum(bool(value) for value in (args.handoff, args.resume_latest, args.import_package)) > 1:
        parser.error(
            "--handoff, --resume-latest, and --import-package are mutually exclusive"
        )
    handoff_path = args.handoff
    if args.resume_latest:
        from picture_tool.gui.operator_handoff import (
            OperatorHandoffError,
            resolve_latest_operator_handoff,
        )

        try:
            handoff_path = str(resolve_latest_operator_handoff(Path.cwd()))
        except OperatorHandoffError as exc:
            parser.exit(2, f"ERROR: {exc}\n")
    if args.import_package:
        from picture_tool.portable_training_package import (
            PortableTrainingImportError,
            import_portable_training_package,
        )

        try:
            imported = import_portable_training_package(
                args.import_package,
                Path.cwd(),
            )
            handoff_path = str(imported.handoff_path)
        except (OSError, zipfile.BadZipFile, PortableTrainingImportError) as exc:
            parser.exit(2, f"ERROR: {exc}\n")
    app = QApplication([sys.argv[0], *qt_args])
    
    # Set Global Font
    font = QtGui.QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Launch Main Window
    window = window_class()
    window.set_background_mode(args.background)
    if args.background:
        if not handoff_path:
            parser.error("--background requires an operator handoff")
        app.setQuitOnLastWindowClosed(False)
    else:
        window.show()
    if handoff_path:
        QTimer.singleShot(
            0, lambda path=handoff_path: window.apply_operator_handoff(path)
        )
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
