from pathlib import Path

from picture_tool import doctor


def test_doctor_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, "_check_import", lambda name: (True, ""))
    monkeypatch.setattr(doctor, "_check_command", lambda cmd: (True, ""))

    code = doctor.run_doctor(create_demo=True)
    assert code == 0
    demo = Path("data/demo_doctor")
    assert (demo / "images").exists()
    assert (demo / "labels").exists()
