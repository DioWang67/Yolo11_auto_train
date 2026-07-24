from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

import picture_tool.runtime_pair_deployment as runtime_pair_module
from picture_tool.runtime_pair_deployment import (
    NumericalComparison,
    PairVerification,
    RuntimePairError,
    deploy_runtime_pair,
    main,
    validate_export_contract,
    verify_runtime_pair,
)


def _passing_comparison(
    _runtime: Path,
    _training: Path,
    _input_size: int,
    _rtol: float,
    _atol: float,
) -> NumericalComparison:
    return NumericalComparison(
        runtime_shape=(1, 9, 8400),
        training_shape=(1, 9, 8400),
        max_abs_error=0.0008,
        mean_abs_error=0.00001,
        p99_abs_error=0.00015,
        passed=True,
        class_names=("Black", "Green", "Orange", "Red", "Yellow"),
    )


def _create_pair(tmp_path: Path) -> tuple[Path, Path]:
    runtime = tmp_path / "best.onnx"
    training = tmp_path / "best.pt"
    runtime.write_bytes(b"onnx-runtime")
    training.write_bytes(b"pt-training")
    return runtime, training


def _contract_payload(runtime: Path, training: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "runtime_format": "onnx",
        "runtime_file": "weights/best.onnx",
        "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "training_weight_file": "weights/best.pt",
        "training_weight_sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
    }


def test_verify_runtime_pair_returns_immutable_identity(tmp_path: Path) -> None:
    runtime, training = _create_pair(tmp_path)

    result = verify_runtime_pair(
        runtime,
        training,
        comparison_runner=_passing_comparison,
    )

    assert result.runtime_path == runtime.resolve()
    assert result.training_weight_path == training.resolve()
    assert result.runtime_sha256 == hashlib.sha256(runtime.read_bytes()).hexdigest()
    assert result.comparison.class_names[2] == "Orange"


def test_verify_runtime_pair_rejects_numerical_mismatch(tmp_path: Path) -> None:
    runtime, training = _create_pair(tmp_path)

    def mismatch(*_args: object) -> NumericalComparison:
        return NumericalComparison(
            runtime_shape=(1, 9, 8400),
            training_shape=(1, 9, 8400),
            max_abs_error=4.2,
            mean_abs_error=0.8,
            p99_abs_error=2.1,
            passed=False,
        )

    with pytest.raises(RuntimePairError, match="numerical outputs do not match"):
        verify_runtime_pair(runtime, training, comparison_runner=mismatch)


def test_verify_runtime_pair_rejects_file_changed_during_probe(tmp_path: Path) -> None:
    runtime, training = _create_pair(tmp_path)

    def mutating_probe(*_args: object) -> NumericalComparison:
        training.write_bytes(b"replaced-while-verifying")
        return _passing_comparison(runtime, training, 640, 0.001, 0.001)

    with pytest.raises(RuntimePairError, match="PT file changed"):
        verify_runtime_pair(runtime, training, comparison_runner=mutating_probe)


@pytest.mark.parametrize(
    ("runtime_name", "training_name", "runtime_bytes", "training_bytes", "message"),
    [
        ("best.pt", "best.pt", b"onnx", b"pt", "Runtime model must use"),
        ("best.onnx", "best.onnx", b"onnx", b"pt", "checkpoint must use"),
        ("best.onnx", "best.pt", b"", b"pt", "Runtime model is empty"),
        ("best.onnx", "best.pt", b"onnx", b"", "checkpoint is empty"),
    ],
)
def test_verify_runtime_pair_rejects_invalid_artifacts(
    tmp_path: Path,
    runtime_name: str,
    training_name: str,
    runtime_bytes: bytes,
    training_bytes: bytes,
    message: str,
) -> None:
    runtime = tmp_path / runtime_name
    training = tmp_path / training_name
    runtime.write_bytes(runtime_bytes)
    training.write_bytes(training_bytes)

    with pytest.raises(RuntimePairError, match=message):
        verify_runtime_pair(runtime, training, comparison_runner=_passing_comparison)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_size": 0}, "Input size"),
        ({"rtol": -0.1}, "tolerances"),
        ({"atol": -0.1}, "tolerances"),
    ],
)
def test_verify_runtime_pair_validates_probe_settings(
    tmp_path: Path, kwargs: dict[str, float | int], message: str
) -> None:
    runtime, training = _create_pair(tmp_path)

    with pytest.raises(RuntimePairError, match=message):
        verify_runtime_pair(
            runtime,
            training,
            comparison_runner=_passing_comparison,
            **kwargs,
        )


def test_verify_runtime_pair_wraps_probe_errors_and_shape_mismatch(
    tmp_path: Path,
) -> None:
    runtime, training = _create_pair(tmp_path)

    def broken_probe(*_args: object) -> NumericalComparison:
        raise OSError("backend unavailable")

    with pytest.raises(RuntimePairError, match="Unable to compare"):
        verify_runtime_pair(runtime, training, comparison_runner=broken_probe)

    def wrong_shape(*_args: object) -> NumericalComparison:
        return NumericalComparison(
            runtime_shape=(1, 9, 8400),
            training_shape=(1, 8, 8400),
            max_abs_error=1.0,
            mean_abs_error=1.0,
            p99_abs_error=1.0,
            passed=False,
        )

    with pytest.raises(RuntimePairError, match="output shapes differ"):
        verify_runtime_pair(runtime, training, comparison_runner=wrong_shape)


def test_verify_runtime_pair_rejects_runtime_changed_during_probe(
    tmp_path: Path,
) -> None:
    runtime, training = _create_pair(tmp_path)

    def mutating_probe(*_args: object) -> NumericalComparison:
        runtime.write_bytes(b"changed-runtime")
        return _passing_comparison(runtime, training, 640, 0.001, 0.001)

    with pytest.raises(RuntimePairError, match="ONNX file changed"):
        verify_runtime_pair(runtime, training, comparison_runner=mutating_probe)


def test_validate_export_contract_checks_both_hashes(tmp_path: Path) -> None:
    runtime, training = _create_pair(tmp_path)
    verification = verify_runtime_pair(
        runtime,
        training,
        comparison_runner=_passing_comparison,
    )
    contract = tmp_path / "runtime_export_manifest.json"
    contract.write_text(
        json.dumps(_contract_payload(runtime, training)), encoding="utf-8"
    )

    payload = validate_export_contract(contract, verification)
    assert payload["runtime_format"] == "onnx"

    corrupted = _contract_payload(runtime, training)
    corrupted["training_weight_sha256"] = "0" * 64
    contract.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(RuntimePairError, match="PT checksum differs"):
        validate_export_contract(contract, verification)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "JSON object"),
        ({"schema_version": 99}, "Unsupported"),
        ({"schema_version": 1, "runtime_format": "engine"}, "does not describe"),
        (
            {
                "schema_version": 1,
                "runtime_format": "onnx",
                "runtime_sha256": "bad",
                "training_weight_sha256": "0" * 64,
            },
            "invalid ONNX checksum",
        ),
        (
            {
                "schema_version": 1,
                "runtime_format": "onnx",
                "runtime_sha256": "0" * 64,
                "training_weight_sha256": "bad",
            },
            "invalid PT checksum",
        ),
    ],
)
def test_validate_export_contract_rejects_invalid_contracts(
    tmp_path: Path, payload: object, message: str
) -> None:
    runtime, training = _create_pair(tmp_path)
    verification = verify_runtime_pair(
        runtime,
        training,
        comparison_runner=_passing_comparison,
    )
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimePairError, match=message):
        validate_export_contract(contract, verification)


def test_validate_export_contract_rejects_missing_malformed_and_runtime_mismatch(
    tmp_path: Path,
) -> None:
    runtime, training = _create_pair(tmp_path)
    verification = verify_runtime_pair(
        runtime,
        training,
        comparison_runner=_passing_comparison,
    )
    contract = tmp_path / "contract.json"
    with pytest.raises(RuntimePairError, match="not found"):
        validate_export_contract(contract, verification)

    contract.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimePairError, match="invalid"):
        validate_export_contract(contract, verification)

    payload = _contract_payload(runtime, training)
    payload["runtime_sha256"] = "0" * 64
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimePairError, match="ONNX checksum differs"):
        validate_export_contract(contract, verification)


def test_deploy_runtime_pair_publishes_versioned_contract(tmp_path: Path) -> None:
    runtime, training = _create_pair(tmp_path)
    verification = verify_runtime_pair(
        runtime,
        training,
        comparison_runner=_passing_comparison,
    )
    models_dir = tmp_path / "inference" / "models"
    station = models_dir / "Cable1" / "A" / "yolo"
    station.mkdir(parents=True)
    (station / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "weights": "models/Cable1/A/yolo/weights/best.onnx",
                "color_score_threshold": 0.4,
                "exposure_time": "29496.0000",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    deployed = deploy_runtime_pair(
        verification,
        inference_models_dir=models_dir,
        product="Cable1",
        area="A",
        export_contract=_contract_payload(runtime, training),
    )

    assert deployed.version == "1.0.0"
    assert deployed.runtime_path.read_bytes() == runtime.read_bytes()
    assert deployed.training_weight_path.read_bytes() == training.read_bytes()
    manifest = yaml.safe_load(deployed.manifest_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(deployed.config_path.read_text(encoding="utf-8"))
    assert manifest["deployed_file"] == deployed.runtime_path.name
    assert manifest["training_weight_file"] == deployed.training_weight_path.name
    assert manifest["weight_sha256"] == verification.runtime_sha256
    assert manifest["training_weight_sha256"] == verification.training_weight_sha256
    assert manifest["pair_verification"]["method"] == (
        "export_contract_and_numerical_equivalence"
    )
    assert config["weights"].endswith(deployed.runtime_path.name)
    assert config["color_score_threshold"] == 0.4
    assert config["exposure_time"] == "29496.0000"
    snapshot = station / manifest["config_snapshot"]
    assert snapshot.is_file()
    assert not (station / ".deploy.lock").exists()

    deployed_again = deploy_runtime_pair(
        verification,
        inference_models_dir=models_dir,
        product="Cable1",
        area="A",
    )
    assert deployed_again.version == "1.0.1"


def test_deploy_runtime_pair_rejects_unsafe_target(tmp_path: Path) -> None:
    runtime, training = _create_pair(tmp_path)
    verification = verify_runtime_pair(
        runtime,
        training,
        comparison_runner=_passing_comparison,
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    with pytest.raises(RuntimePairError, match="Invalid product"):
        deploy_runtime_pair(
            verification,
            inference_models_dir=models_dir,
            product="../Cable1",
            area="A",
        )


def test_deploy_runtime_pair_validates_target_and_station_config(tmp_path: Path) -> None:
    runtime, training = _create_pair(tmp_path)
    verification = verify_runtime_pair(
        runtime,
        training,
        comparison_runner=_passing_comparison,
    )
    missing_models = tmp_path / "missing-models"
    with pytest.raises(RuntimePairError, match="models directory was not found"):
        deploy_runtime_pair(
            verification,
            inference_models_dir=missing_models,
            product="Cable1",
            area="A",
        )

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    with pytest.raises(RuntimePairError, match="Station config was not found"):
        deploy_runtime_pair(
            verification,
            inference_models_dir=models_dir,
            product="Cable1",
            area="A",
        )

    station = models_dir / "Cable1" / "A" / "yolo"
    station.mkdir(parents=True)
    (station / "config.yaml").write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(RuntimePairError, match="YAML mapping"):
        deploy_runtime_pair(
            verification,
            inference_models_dir=models_dir,
            product="Cable1",
            area="A",
        )


def test_deploy_runtime_pair_rejects_changed_source_and_lock_timeout(
    tmp_path: Path,
) -> None:
    runtime, training = _create_pair(tmp_path)
    verification = verify_runtime_pair(
        runtime,
        training,
        comparison_runner=_passing_comparison,
    )
    models_dir = tmp_path / "models"
    station = models_dir / "Cable1" / "A" / "yolo"
    station.mkdir(parents=True)
    (station / "config.yaml").write_text("weights: old.onnx\n", encoding="utf-8")

    runtime.write_bytes(b"changed-after-verification")
    with pytest.raises(RuntimePairError, match="ONNX source changed"):
        deploy_runtime_pair(
            verification,
            inference_models_dir=models_dir,
            product="Cable1",
            area="A",
        )

    runtime.write_bytes(b"onnx-runtime")
    (station / ".deploy.lock").mkdir()
    with pytest.raises(RuntimePairError, match="Timed out"):
        deploy_runtime_pair(
            verification,
            inference_models_dir=models_dir,
            product="Cable1",
            area="A",
            lock_timeout=0,
        )


def test_deploy_runtime_pair_rolls_back_when_manifest_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, training = _create_pair(tmp_path)
    verification = verify_runtime_pair(
        runtime,
        training,
        comparison_runner=_passing_comparison,
    )
    models_dir = tmp_path / "models"
    station = models_dir / "Cable1" / "A" / "yolo"
    station.mkdir(parents=True)
    config = station / "config.yaml"
    original = b"weights: old.onnx\ncolor_score_threshold: 0.4\n"
    config.write_bytes(original)
    real_write = runtime_pair_module._write_yaml_atomic

    def fail_manifest(path: Path, payload: dict[str, object]) -> None:
        if path.name == "deployment_manifest.yaml":
            raise OSError("disk full")
        real_write(path, payload)

    monkeypatch.setattr(runtime_pair_module, "_write_yaml_atomic", fail_manifest)
    with pytest.raises(RuntimePairError, match="Unable to publish"):
        deploy_runtime_pair(
            verification,
            inference_models_dir=models_dir,
            product="Cable1",
            area="A",
        )

    assert config.read_bytes() == original
    assert not (station / "deployment_manifest.yaml").exists()
    assert not list((station / "weights").glob("Cable1_A_v*.onnx"))
    assert not (station / ".deploy.lock").exists()


def test_ordered_class_names_supports_mapping_sequence_and_invalid_values() -> None:
    assert runtime_pair_module._ordered_class_names({"1": "Green", "0": "Black"}) == (
        "Black",
        "Green",
    )
    assert runtime_pair_module._ordered_class_names(["Black", "Green"]) == (
        "Black",
        "Green",
    )
    assert runtime_pair_module._ordered_class_names({"bad": "Black"}) == ()
    assert runtime_pair_module._ordered_class_names("Black") == ()


def test_main_reports_validation_only_and_expected_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime, training = _create_pair(tmp_path)
    verification = verify_runtime_pair(
        runtime,
        training,
        comparison_runner=_passing_comparison,
    )
    monkeypatch.setattr(runtime_pair_module, "verify_runtime_pair", lambda *_a, **_k: verification)

    assert main(["--weights", str(runtime), "--training-weights", str(training)]) == 0
    assert "Validation only" in capsys.readouterr().out

    def reject(*_args: object, **_kwargs: object) -> PairVerification:
        raise RuntimePairError("pair rejected")

    monkeypatch.setattr(runtime_pair_module, "verify_runtime_pair", reject)
    assert main(["--weights", str(runtime), "--training-weights", str(training)]) == 2
    assert "pair rejected" in capsys.readouterr().out
