"""Compatibility stamp written into every Anomalib model package.

The inference station loads ``weights/model.ckpt`` with whatever Anomalib it
has installed. A checkpoint from another Anomalib major, a different resize,
or a silently changed default pre-processor would still load and still score,
just differently. The stamp lets the station refuse such a model before it
judges a part.

The stamp records the runtime that produced the checkpoint, the checkpoint
hash, and the raw score of a probe image. The probe is a closed-form pattern
rather than seeded noise: ``numpy.random.Generator`` streams are not guaranteed
across numpy versions, and a probe that changed with numpy would reject good
models. The probe parameters travel inside the stamp so the station rebuilds
the identical image without sharing code with this repository.

The raw score is taken before the post-processor. The normalised score
saturates at 0 or 1 on out-of-distribution inputs and would not move when the
model or its pre-processing changes.
"""

from __future__ import annotations

import hashlib
import importlib
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_SCHEMA = 2
PROBE_KIND = "sincos_v2"
PROBE_BASE = 0.45
PROBE_AMPLITUDE = 0.05
PROBE_ROW_FREQUENCY = 0.11
PROBE_COLUMN_FREQUENCY = 0.07
PROBE_CHANNEL_PHASE = 0.7
PROBE_DETAIL_AMPLITUDE = 0.04
PROBE_DETAIL_FREQUENCY = 1.3
PROBE_ABS_TOLERANCE = 1e-4
PROBE_REL_TOLERANCE = 1e-3


def probe_size(image_size: int) -> int:
    """Probe side length: deliberately not the model size, so the resize runs."""
    return image_size * 3 // 2 + 7


def probe_image(image_size: int) -> np.ndarray:
    """Return the ``(3, s, s)`` float32 probe image in ``[0, 1]``, s = probe_size.

    value = base + amplitude * sin(row_f * i + phase * c) * cos(column_f * j)
                 + detail_amplitude * sin(detail_f * i) * sin(detail_f * j)

    The probe is larger than the model input and carries near-Nyquist detail,
    so the pre-processor actually resamples it. A change of resize method or
    antialiasing then moves the score, which is what the stamp must catch.
    """
    size = probe_size(image_size)
    rows = np.arange(size, dtype=np.float64)[None, :, None]
    columns = np.arange(size, dtype=np.float64)[None, None, :]
    channels = np.arange(3, dtype=np.float64)[:, None, None]
    values = (
        PROBE_BASE
        + PROBE_AMPLITUDE
        * np.sin(PROBE_ROW_FREQUENCY * rows + PROBE_CHANNEL_PHASE * channels)
        * np.cos(PROBE_COLUMN_FREQUENCY * columns)
        + PROBE_DETAIL_AMPLITUDE
        * np.sin(PROBE_DETAIL_FREQUENCY * rows)
        * np.sin(PROBE_DETAIL_FREQUENCY * columns)
    )
    return values.astype(np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(class_path: str, checkpoint_path: Path) -> Any:
    """Load an Anomalib 2.x model for inference, refusing pickled code.

    ``weights_only=True`` is passed explicitly: these files are copied to
    stations, and a checkpoint that needs arbitrary unpickling (every
    Anomalib 1.x checkpoint does) must fail here rather than run.
    """
    module_name, _, class_name = class_path.rpartition(".")
    model_class = getattr(importlib.import_module(module_name), class_name)
    model = model_class.load_from_checkpoint(str(checkpoint_path), weights_only=True)
    return model.eval()


def raw_score(model: Any, image_chw: np.ndarray) -> float:
    """Score one ``(3, H, W)`` image the way training-time prediction does.

    ``model(batch)`` would apply ``pre_processor.export_transform`` (resize
    without antialiasing), which scores up to ~0.05 away from what
    ``Engine.predict`` and training produce. The training-time transform is
    applied explicitly instead, and the score is read before post-processing.
    """
    import torch

    with torch.inference_mode():
        batch = torch.from_numpy(np.ascontiguousarray(image_chw))[None]
        transformed = model.pre_processor.transform(batch)
        output = model.model(transformed)
    return float(output.pred_score.reshape(-1)[0])


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def compute_stamp(
    checkpoint_path: Path,
    *,
    class_path: str,
    image_size: int,
) -> dict[str, Any]:
    """Build the compatibility stamp for a trained checkpoint."""
    model = load_model(class_path, checkpoint_path)
    score = raw_score(model, probe_image(image_size))
    return {
        "artifact_schema": ARTIFACT_SCHEMA,
        "runtime": {
            "anomalib": _version("anomalib"),
            "torch": _version("torch"),
            "lightning": _version("lightning"),
            "python": platform.python_version(),
        },
        "class_path": class_path,
        "image_size": int(image_size),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "probe": {
            "kind": PROBE_KIND,
            "base": PROBE_BASE,
            "amplitude": PROBE_AMPLITUDE,
            "row_frequency": PROBE_ROW_FREQUENCY,
            "column_frequency": PROBE_COLUMN_FREQUENCY,
            "channel_phase": PROBE_CHANNEL_PHASE,
            "detail_amplitude": PROBE_DETAIL_AMPLITUDE,
            "detail_frequency": PROBE_DETAIL_FREQUENCY,
            "size": probe_size(image_size),
            "raw_pred_score": score,
            "abs_tolerance": PROBE_ABS_TOLERANCE,
            "rel_tolerance": PROBE_REL_TOLERANCE,
        },
    }
