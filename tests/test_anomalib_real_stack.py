"""Train, package and reload a tiny real Anomalib model end to end.

Every other Anomalib test replaces training with a stub, and
test_color_integration swaps torch for a MagicMock for the rest of the
session. This one runs in a fresh interpreter so the real stack is exercised:
Folder datamodule, model pre-processor, Engine.fit, the compatibility stamp,
and a weights_only reload that reproduces the stamped probe score.
"""

import json
import subprocess
import sys
import textwrap
from importlib.util import find_spec

import pytest

pytestmark = pytest.mark.skipif(
    find_spec("anomalib") is None, reason="anomalib is not installed"
)

SCRIPT = textwrap.dedent(
    """
    import json, sys, warnings
    from pathlib import Path
    warnings.filterwarnings("ignore")
    import cv2, numpy as np
    from picture_tool.train import anomalib_compat, anomalib_trainer

    root = Path(sys.argv[1])
    normal = root / "data" / "train" / "good"
    normal.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(10):
        image = np.full((64, 64, 3), 120, np.uint8)
        image += rng.integers(0, 12, image.shape, dtype=np.uint8)
        cv2.imwrite(str(normal / f"{index}.png"), image)

    trained = anomalib_trainer.train_anomalib_folder(
        root / "data", product="P", area="A", project=root / "runs",
        model="padim", image_size=64, batch_size=4, max_epochs=1,
        accelerator="cpu", devices=1, pre_trained=False, tmp_dir=root / "tmp",
    )
    packaged = anomalib_trainer.package_anomalib_run(
        trained.run_dir, output_dir=root / "packages", product="P", area="A"
    )
    stamp = json.loads(packaged.manifest_path.read_text(encoding="utf-8"))["compatibility"]
    model = anomalib_compat.load_model(stamp["class_path"], packaged.checkpoint_path)
    rescored = anomalib_compat.raw_score(model, anomalib_compat.probe_image(stamp["image_size"]))
    print(json.dumps({"stamp": stamp, "rescored": rescored}))
    """
)


def test_padim_trains_packages_and_reloads_with_its_stamp(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    stamp = result["stamp"]

    assert stamp["artifact_schema"] == 2
    assert stamp["class_path"] == "anomalib.models.Padim"
    assert stamp["image_size"] == 64
    assert stamp["runtime"]["anomalib"].startswith("2.")
    assert len(stamp["checkpoint_sha256"]) == 64
    probe = stamp["probe"]
    assert probe["size"] != stamp["image_size"]  # the resize must actually run
    assert abs(result["rescored"] - probe["raw_pred_score"]) <= probe["abs_tolerance"]
