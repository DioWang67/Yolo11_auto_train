
from unittest.mock import patch, MagicMock
from picture_tool.utils.dvc_wrapper import DVCWrapper
import shutil

@patch("shutil.which")
def test_dvc_not_installed(mock_which):
    mock_which.return_value = None
    dvc = DVCWrapper()
    assert not dvc.is_installed
    assert not dvc.pull()

@patch("shutil.which")
@patch("subprocess.run")
def test_dvc_pull(mock_run, mock_which):
    mock_which.return_value = "/usr/bin/dvc"
    dvc = DVCWrapper()
    
    assert dvc.is_installed
    
    # Test valid pull
    mock_run.return_value.returncode = 0
    assert dvc.pull()
    mock_run.assert_called_with(["dvc", "pull"], cwd=dvc.cwd, check=True, capture_output=False)

@patch("shutil.which")
@patch("subprocess.run")
def test_dvc_fail(mock_run, mock_which):
    mock_which.return_value = "/usr/bin/dvc"
    dvc = DVCWrapper()
    
    # force exception
    mock_run.side_effect = Exception("Boom")
    assert not dvc.pull()
