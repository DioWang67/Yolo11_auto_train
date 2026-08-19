from __future__ import annotations

from pathlib import Path

import pytest

from scripts.migrate_workspace_paths import MigrationError, main, migrate_workspace_paths


def _roots(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    old_training = (tmp_path / "legacy" / "training-project").resolve()
    old_inference = (tmp_path / "legacy" / "inference-project").resolve()
    new_training = (tmp_path / "workspace" / "training-project").resolve()
    new_inference = (tmp_path / "workspace" / "inference-project").resolve()
    new_training.mkdir(parents=True)
    new_inference.mkdir(parents=True)
    return old_training, new_training, old_inference, new_inference


def _windows(path: Path) -> bytes:
    return str(path).replace("/", "\\").encode()


def _posix(path: Path) -> bytes:
    return str(path).replace("\\", "/").encode()


def _escaped_windows(path: Path) -> bytes:
    return _windows(path).replace(b"\\", b"\\\\")


def test_dry_run_audits_without_writing_and_skips_non_text_and_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old_train, new_train, old_infer, new_infer = _roots(tmp_path)
    config = new_train / "config.yaml"
    original = (
        b'train: "'
        + _windows(old_train)
        + b'\\runs"\r\ninfer: "'
        + _posix(old_infer)
        + b'/models"\r\n'
    )
    config.write_bytes(original)
    (new_train / "model.onnx").write_bytes(_windows(old_train))
    cache_file = new_infer / ".cache" / "state.json"
    cache_file.parent.mkdir()
    cache_file.write_bytes(_posix(old_infer))

    audit = migrate_workspace_paths(
        old_training_root=old_train,
        new_training_root=new_train,
        old_inference_root=old_infer,
        new_inference_root=new_infer,
    )

    assert audit.scanned_files == 1
    assert audit.matched_files == 1
    assert audit.replacement_count == 2
    assert audit.updated_files == 0
    assert config.read_bytes() == original
    assert (new_train / "model.onnx").read_bytes() == _windows(old_train)
    assert cache_file.read_bytes() == _posix(old_infer)
    output = capsys.readouterr().out
    assert "Mode: DRY-RUN" in output
    assert "WOULD UPDATE: training:config.yaml" in output
    assert "No files were written" in output


def test_apply_preserves_other_bytes_and_is_idempotent(tmp_path: Path) -> None:
    old_train, new_train, old_infer, new_infer = _roots(tmp_path)
    training_file = new_train / "paths.txt"
    inference_file = new_infer / "runtime.json"
    sibling_text = _windows(old_train) + b"-backup"
    training_file.write_bytes(
        b"\xffprefix\r\n"
        + _windows(old_train)
        + b"\\configs\r\n"
        + sibling_text
        + b"\r\n"
    )
    inference_file.write_bytes(
        b'{"root":"'
        + _escaped_windows(old_infer)
        + b'\\\\models","alt":"'
        + _posix(old_train)
        + b'/runs"}\r\n'
    )

    first = migrate_workspace_paths(
        old_training_root=old_train,
        new_training_root=new_train,
        old_inference_root=old_infer,
        new_inference_root=new_infer,
        apply=True,
        emit_audit=False,
    )

    assert first.matched_files == 2
    assert first.replacement_count == 3
    assert first.updated_files == 2
    assert training_file.read_bytes() == (
        b"\xffprefix\r\n"
        + _windows(new_train)
        + b"\\configs\r\n"
        + sibling_text
        + b"\r\n"
    )
    assert inference_file.read_bytes() == (
        b'{"root":"'
        + _escaped_windows(new_infer)
        + b'\\\\models","alt":"'
        + _posix(new_train)
        + b'/runs"}\r\n'
    )

    bytes_after_first_apply = (
        training_file.read_bytes(),
        inference_file.read_bytes(),
    )
    second = migrate_workspace_paths(
        old_training_root=old_train,
        new_training_root=new_train,
        old_inference_root=old_infer,
        new_inference_root=new_infer,
        apply=True,
        emit_audit=False,
    )

    assert second.matched_files == 0
    assert second.replacement_count == 0
    assert second.updated_files == 0
    assert (training_file.read_bytes(), inference_file.read_bytes()) == bytes_after_first_apply


def test_cli_requires_apply_to_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old_train, new_train, old_infer, new_infer = _roots(tmp_path)
    config = new_infer / "settings.py"
    config.write_bytes(b"ROOT = r'" + _windows(old_train) + b"\\models'")
    arguments = [
        "--old-training-root",
        str(old_train),
        "--new-training-root",
        str(new_train),
        "--old-inference-root",
        str(old_infer),
        "--new-inference-root",
        str(new_infer),
    ]

    assert main(arguments) == 0
    assert _windows(old_train) in config.read_bytes()
    assert "Mode: DRY-RUN" in capsys.readouterr().out

    assert main([*arguments, "--apply"]) == 0
    assert _windows(new_train) in config.read_bytes()
    assert _windows(old_train) not in config.read_bytes()
    assert "Files atomically updated: 1" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"old_training_root": "relative/train"}, "must be an absolute path"),
        ({"new_inference_root": "missing"}, "must be an absolute path"),
    ],
)
def test_invalid_root_input_is_refused(
    tmp_path: Path, override: dict[str, str], message: str
) -> None:
    old_train, new_train, old_infer, new_infer = _roots(tmp_path)
    arguments: dict[str, str | Path] = {
        "old_training_root": old_train,
        "new_training_root": new_train,
        "old_inference_root": old_infer,
        "new_inference_root": new_infer,
    }
    arguments.update(override)

    with pytest.raises(MigrationError, match=message):
        migrate_workspace_paths(**arguments, emit_audit=False)


def test_missing_or_overlapping_new_roots_are_refused(tmp_path: Path) -> None:
    old_train, new_train, old_infer, new_infer = _roots(tmp_path)
    missing_root = (tmp_path / "workspace" / "missing").resolve()

    with pytest.raises(MigrationError, match="does not exist"):
        migrate_workspace_paths(
            old_training_root=old_train,
            new_training_root=new_train,
            old_inference_root=old_infer,
            new_inference_root=missing_root,
            emit_audit=False,
        )

    nested_inference = new_train / "nested-inference"
    nested_inference.mkdir()
    with pytest.raises(MigrationError, match="must not overlap"):
        migrate_workspace_paths(
            old_training_root=old_train,
            new_training_root=new_train,
            old_inference_root=old_infer,
            new_inference_root=nested_inference,
            emit_audit=False,
        )


def test_new_root_inside_old_root_is_refused_for_idempotency(tmp_path: Path) -> None:
    old_training = (tmp_path / "old-training").resolve()
    new_training = old_training / "moved"
    new_training.mkdir(parents=True)
    old_inference = (tmp_path / "old-inference").resolve()
    new_inference = (tmp_path / "workspace" / "inference").resolve()
    new_inference.mkdir(parents=True)

    with pytest.raises(MigrationError, match="non-idempotent"):
        migrate_workspace_paths(
            old_training_root=old_training,
            new_training_root=new_training,
            old_inference_root=old_inference,
            new_inference_root=new_inference,
            emit_audit=False,
        )
