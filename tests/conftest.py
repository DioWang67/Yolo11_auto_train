
import pytest
import tqdm
import matplotlib

@pytest.fixture(autouse=True, scope="session")
def configure_env():
    # Disable tqdm monitor thread to prevent segfaults in CI
    tqdm.monitor_interval = 0
    
    # Use non-interactive backend for matplotlib
    matplotlib.use("Agg")
    
    yield
