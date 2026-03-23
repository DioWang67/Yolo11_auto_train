import pytest
import os
from unittest.mock import MagicMock, patch
from picture_tool.tracking.experiment_tracker import (
    MLflowTracker,
    TrackingInfrastructureError,
    TrackingDomainError,
)

@patch("picture_tool.tracking.experiment_tracker.mlflow")
def test_mlflow_tracker_infrastructure_error_init(mock_mlflow):
    """Test that MLflowTracker raises infrastructure err instead of silently failing."""
    from picture_tool.tracking.experiment_tracker import MlflowException
    
    mock_mlflow.set_tracking_uri.side_effect = MlflowException("Server down")
    
    with pytest.raises(TrackingInfrastructureError, match="Failed to setup MLflow"):
        MLflowTracker(tracking_uri="http://bad-uri")

@patch("picture_tool.tracking.experiment_tracker.mlflow")
def test_mlflow_tracker_domain_error_artifact(mock_mlflow):
    """Test that missing artifact throws a domain error."""
    tracker = MLflowTracker(tracking_uri="http://good-uri")
    
    with pytest.raises(TrackingDomainError, match="Artifact not found"):
        tracker.log_artifact("/path/to/nonexistent/file.txt")
        
@patch("picture_tool.tracking.experiment_tracker.mlflow")
def test_mlflow_tracker_infrastructure_error_log(mock_mlflow):
    """Test that failing to log params throws infrastructure error."""
    from picture_tool.tracking.experiment_tracker import MlflowException
    
    tracker = MLflowTracker(tracking_uri="http://good-uri")
    mock_mlflow.log_params.side_effect = MlflowException("Network Timeout")
    
    with pytest.raises(TrackingInfrastructureError, match="MLflow log_params failed"):
        tracker.log_params({"lr": 0.01})
