"""Independent opinions about what is in a box.

A detector labelling its own output is not evidence, it is a restatement.
Whatever confidence it attaches, the class it chose and the confidence it
reports come from the same weights, so a systematic error --- calling Red
Orange, say --- arrives already agreed with itself. Pseudo-labels built that
way inherit the model's mistakes and then train them back in, which is worse
than having less data.

So a proposal here is assembled from sources that can *disagree*. The
detector proposes boxes and classes. The colour measurement re-derives the
class of each box from pixels through the project's existing per-colour
strategies, which know nothing about the detector. Where they agree, that is
worth something; where they disagree on a pair the station actually confuses,
the sample goes to a person.

``EvidenceSource`` is a protocol rather than a base class, and the composer
takes a list of them, because the useful sources are not all available yet.
A foundation detector or a segmentation model would slot in here as another
opinion; none is installed in this environment, so none is implemented, and
nothing about the design assumes there are only two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from picture_tool.autotrain import AutoTrainError
from picture_tool.bootstrap.profile import ProductProfile


class EvidenceError(AutoTrainError):
    """Raised when an evidence source cannot run at all."""


@dataclass(frozen=True)
class Box:
    """One candidate object, in normalised xywh so it is resolution-free."""

    class_name: str
    cx: float
    cy: float
    width: float
    height: float
    confidence: float
    source: str = ""

    def to_pixels(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        half_w = self.width * image_width / 2.0
        half_h = self.height * image_height / 2.0
        centre_x = self.cx * image_width
        centre_y = self.cy * image_height
        return (
            max(0, int(round(centre_x - half_w))),
            max(0, int(round(centre_y - half_h))),
            min(image_width, int(round(centre_x + half_w))),
            min(image_height, int(round(centre_y + half_h))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "cx": round(self.cx, 6),
            "cy": round(self.cy, 6),
            "width": round(self.width, 6),
            "height": round(self.height, 6),
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }


@dataclass(frozen=True)
class BoxOpinion:
    """One source's reading of one box."""

    source: str
    class_name: str
    confidence: float
    detail: str = ""
    scores: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "detail": self.detail,
            "scores": {k: round(v, 4) for k, v in sorted(self.scores.items())},
        }


class EvidenceSource(Protocol):
    """Something that can say what a box contains."""

    name: str

    def read(
        self, image: np.ndarray, boxes: Sequence[Box], profile: ProductProfile
    ) -> list[BoxOpinion]:
        """Return one opinion per box, in the same order."""


# ---------------------------------------------------------------------------
# Detector


class DetectorProposer:
    """Proposes the boxes themselves, using the station's deployed detector.

    Only the *geometry* is taken on trust here. The detector's class is
    carried through as one opinion among others rather than as the answer,
    which is the difference between using a model to find objects and using
    it to label them.
    """

    name = "detector"

    def __init__(
        self,
        weights: str | Path,
        *,
        confidence: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
        predictor: Any = None,
    ) -> None:
        self.weights = str(weights)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self._predictor = predictor

    def _load(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        import os

        if os.environ.get("PYTEST_IS_RUNNING") == "1":
            raise EvidenceError(
                "Refusing to load ultralytics under pytest; inject a predictor."
            )
        from ultralytics import YOLO

        model = YOLO(self.weights)

        def predict(path: str) -> Any:
            return model.predict(
                path,
                conf=self.confidence,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )

        self._predictor = predict
        return predict

    def propose(self, image_path: str | Path, profile: ProductProfile) -> list[Box]:
        """Boxes the detector found, in normalised coordinates."""
        results = self._load()(str(image_path))
        boxes: list[Box] = []
        for result in results or []:
            names = getattr(result, "names", {}) or {}
            container = getattr(result, "boxes", None)
            if container is None:
                continue
            xywhn = _as_rows(getattr(container, "xywhn", None))
            classes = _as_list(getattr(container, "cls", None))
            confidences = _as_list(getattr(container, "conf", None))
            for index, row in enumerate(xywhn):
                if len(row) != 4:
                    continue
                class_id = _as_int(_at(classes, index))
                name = str(names.get(class_id, "")) if class_id is not None else ""
                if name not in profile.class_schema.names:
                    # A class the station's contract does not contain is not
                    # something to quietly renumber; it is reported and the
                    # sample will fail validation.
                    name = name or f"class_{class_id}"
                boxes.append(
                    Box(
                        class_name=name,
                        cx=float(row[0]),
                        cy=float(row[1]),
                        width=float(row[2]),
                        height=float(row[3]),
                        confidence=float(_at(confidences, index) or 0.0),
                        source=self.name,
                    )
                )
        return boxes

    def read(
        self, image: np.ndarray, boxes: Sequence[Box], profile: ProductProfile
    ) -> list[BoxOpinion]:
        return [
            BoxOpinion(
                source=self.name,
                class_name=box.class_name,
                confidence=box.confidence,
                detail="detector class head",
            )
            for box in boxes
        ]


# ---------------------------------------------------------------------------
# Colour


class ColourEvidence:
    """Re-derives each box's class from its pixels.

    Reuses the project's existing per-colour strategies rather than a new
    colour rule. They are already tuned against this station's reference
    statistics, they already contain the Orange/Red tiebreak that exists
    because those two are what the line confuses, and a second
    implementation would be a second answer to the same question --- the
    inference and training sides of this repository have already been bitten
    by exactly that.

    Knows nothing about the detector, which is what makes its agreement
    worth anything.
    """

    name = "colour"

    def __init__(self, colour_ranges: Mapping[str, Any]) -> None:
        if not colour_ranges:
            raise EvidenceError(
                "No colour reference statistics. Colour evidence without a "
                "reference is just a guess with a number attached."
            )
        from picture_tool.color.strategies.registry import ColorStrategyRegistry

        ColorStrategyRegistry.initialize()
        self._registry = ColorStrategyRegistry
        self._ranges = dict(colour_ranges)

    def read(
        self, image: np.ndarray, boxes: Sequence[Box], profile: ProductProfile
    ) -> list[BoxOpinion]:
        import cv2

        height, width = image.shape[:2]
        inset_x, inset_y = profile.colour_roi_inset
        opinions: list[BoxOpinion] = []
        for box in boxes:
            x1, y1, x2, y2 = box.to_pixels(width, height)
            crop = _inset(image[y1:y2, x1:x2], inset_x, inset_y)
            if crop.size == 0:
                opinions.append(
                    BoxOpinion(
                        source=self.name,
                        class_name="",
                        confidence=0.0,
                        detail="box has no pixels to measure",
                    )
                )
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            scores: dict[str, float] = {}
            for name in profile.class_schema.names:
                colour_range = self._ranges.get(name)
                if colour_range is None:
                    continue
                strategy = self._registry.get_strategy(name)
                try:
                    score, _ = strategy.match_ratio(hsv, lab, colour_range)
                except Exception as exc:  # noqa: BLE001 - opencv raises broadly
                    raise EvidenceError(
                        f"Colour strategy for {name} failed: {exc}"
                    ) from exc
                scores[name] = float(score)
            if not scores:
                opinions.append(
                    BoxOpinion(
                        source=self.name,
                        class_name="",
                        confidence=0.0,
                        detail="no reference statistics for any class",
                    )
                )
                continue
            best = max(scores, key=lambda key: scores[key])
            opinions.append(
                BoxOpinion(
                    source=self.name,
                    class_name=best,
                    confidence=scores[best],
                    detail="measured from pixels, independent of the detector",
                    scores=scores,
                )
            )
        return opinions


def load_colour_ranges(stats_path: str | Path) -> dict[str, Any]:
    """Load the station's reference colour statistics.

    These are the "few reference images" the bootstrapper is given, already
    distilled: production computes them from samples a person confirmed, and
    reusing that file means the bootstrapper and the line agree on what the
    colours are.
    """
    from picture_tool.color.color_verifier import load_color_ranges

    return dict(load_color_ranges(Path(stats_path)))


# ---------------------------------------------------------------------------


def _inset(crop: np.ndarray, inset_x: float, inset_y: float) -> np.ndarray:
    if crop.size == 0:
        return crop
    height, width = crop.shape[:2]
    dx = int(width * max(0.0, min(0.45, inset_x)))
    dy = int(height * max(0.0, min(0.45, inset_y)))
    trimmed = crop[dy : height - dy or height, dx : width - dx or width]
    return trimmed if trimmed.size else crop


def _as_rows(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [list(row) for row in value] if isinstance(value, (list, tuple)) else []


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def _at(values: Sequence[Any], index: int) -> Any:
    return values[index] if 0 <= index < len(values) else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
