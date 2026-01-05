"""
Pytest configuration to prevent CI environment conflicts.
CRITICAL: This must execute before any test imports that use tqdm or matplotlib.
"""
import os
import gc
import pytest
import matplotlib

# MUST be set before importing tqdm anywhere
os.environ["TQDM_DISABLE"] = "1"

# Configure matplotlib to non-interactive backend (must be before pyplot import)
matplotlib.use("Agg")

# Additional safeguard: completely disable tqdm monitor
import tqdm.std
tqdm.std.TMonitor = type('TMonitor', (), {'__init__': lambda *args, **kwargs: None, 'exit': lambda *args: None})
tqdm.tqdm.monitor_interval = 0

@pytest.fixture(scope="session", autouse=True)
def cleanup_resources():
    """Force cleanup of resources after test session to prevent segfaults."""
    yield
    # Force garbage collection before exit
    gc.collect()
    gc.collect()  # Double call to ensure circular references are cleaned
