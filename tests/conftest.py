"""
Pytest configuration to prevent CI environment conflicts.
CRITICAL: This must execute before any test imports that use tqdm or matplotlib.
"""

import os

# Bypass fatal PyTorch/OpenMP DLL conflicts under Windows Pytest
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# We use an explicit environment flag to safely bypass deep C++ imports 
# (PyTorch/Albumentations/CV2) that natively segfault in pytest under Windows Conda.
os.environ["PYTEST_IS_RUNNING"] = "1"
try:
    import matplotlib
    # Configure matplotlib to non-interactive backend (must be before pyplot import)
    matplotlib.use("Agg")
except ImportError:
    pass

# MUST be set before importing tqdm anywhere
os.environ["TQDM_DISABLE"] = "1"

# Additional safeguard: completely disable tqdm monitor
import tqdm.std

tqdm.std.TMonitor = type(
    "TMonitor",
    (),
    {"__init__": lambda *args, **kwargs: None, "exit": lambda *args: None},
)
tqdm.tqdm.monitor_interval = 0

# Disabled explicit gc.collect() to prevent PyTorch DLL teardown segfaults
