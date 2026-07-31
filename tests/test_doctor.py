from unittest.mock import patch
from picture_tool import doctor


def test_check_import_success():
    ok, msg = doctor._check_import("sys")
    assert ok is True
    assert msg == ""


def test_check_import_failure():
    ok, msg = doctor._check_import("non_existent_package_xyz")
    assert ok is False
    assert "No module named" in msg or "not found" in msg


def test_distribution_version_mismatch_is_rejected():
    with patch("importlib.metadata.version", return_value="9.9.9"):
        ok, msg = doctor._check_distribution_version("torch", "2.4.1")
    assert ok is False
    assert "expected=('2.4.1',) actual=9.9.9" in msg


def test_runtime_profile_rejects_mixed_torch_pair():
    versions = {
        "torch": "2.4.1",
        "torchvision": "0.20.1",
        "numpy": "1.26.4",
        "opencv-python": "4.9.0.80",
        "ultralytics": "8.3.156",
    }
    with (
        patch.object(doctor.sys, "version_info", (3, 11, 0)),
        patch("importlib.metadata.version", side_effect=versions.__getitem__),
    ):
        ok, msg = doctor._check_runtime_profile()
    assert ok is False
    assert "unsupported version combination" in msg


def test_check_command_success():
    # echo is available on all platforms (shell=True needed mostly, but doctor uses list args)
    # We can mock subprocess.run for deterministic behavior
    with patch("subprocess.run"):
        ok, msg = doctor._check_command(["dummy", "--version"])
        assert ok is True
        assert msg == ""


def test_check_command_failure():
    import subprocess
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "dummy")):
        ok, msg = doctor._check_command(["dummy"])
        assert ok is False
        assert "exit status 1" in msg


def test_create_demo_dataset(tmp_path):
    root = tmp_path / "doctor_demo"
    doctor._create_demo_dataset(root)

    assert (root / "images").exists()
    assert (root / "labels").exists()
    assert len(list((root / "images").glob("*.jpg"))) == 3
    assert len(list((root / "labels").glob("*.txt"))) == 3


def test_run_doctor_all_ok(capsys):
    # Mock checks to all return True
    with (
        patch("picture_tool.doctor._check_import", return_value=(True, "")),
        patch(
            "picture_tool.doctor._check_distribution_version",
            return_value=(True, ""),
        ),
        patch("picture_tool.doctor._check_runtime_profile", return_value=(True, "")),
        patch("picture_tool.doctor._check_command", return_value=(True, "")),
    ):
        ret = doctor.run_doctor(create_demo=False)
        assert ret == 0

        captured = capsys.readouterr()
        assert "MISSING" not in captured.out
        assert "python" in captured.out


def test_run_doctor_failures(capsys):
    # Mock one clean, one fail
    def mock_import(name):
        if name == "torch":
            return False, "Not installed"
        return True, ""

    with (
        patch("picture_tool.doctor._check_import", side_effect=mock_import),
        patch(
            "picture_tool.doctor._check_distribution_version",
            return_value=(True, ""),
        ),
        patch("picture_tool.doctor._check_runtime_profile", return_value=(True, "")),
        patch("picture_tool.doctor._check_command", return_value=(True, "")),
    ):
        ret = doctor.run_doctor(create_demo=False)
        assert ret == 1

        captured = capsys.readouterr()
        assert "torch       : MISSING - Not installed" in captured.out
        assert "python      : OK" in captured.out
