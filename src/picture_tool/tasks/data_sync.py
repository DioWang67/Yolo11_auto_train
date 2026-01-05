import logging
from typing import Any
from picture_tool.pipeline.core import Task
from picture_tool.utils.dvc_wrapper import DVCWrapper

logger = logging.getLogger(__name__)

def run_data_sync(config: dict, args: Any) -> None:
    """Pull latest data using DVC."""
    dvc = DVCWrapper()
    
    if not dvc.is_installed:
        logger.warning("DVC not installed. Skipping data sync.")
        return

    if not dvc.is_dvc_repo:
        logger.info("Not a DVC repository. Skipping data sync.")
        return

    logger.info("Pulling latest data from DVC remote...")
    if dvc.pull():
        logger.info("DVC pull complete.")
    else:
        # We don't crash the pipeline, maybe they are offline or just testing local changes
        logger.warning("DVC pull failed or no data to pull (are you offline?). Proceeding with local data.")

TASKS = [
    Task(
        name="data_sync",
        run=run_data_sync,
        description="Pull latest dataset versions using DVC (if configured).",
    )
]
