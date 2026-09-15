"""A vision model as an evidence source, and as a cold-start proposer.

Measured on this station before being written, because the model's strengths
here are lopsided and the architecture follows from which is which. On the
six verified frames of 2026-09-15, asked to name and locate all six wire ends:

===========================  =========================================
naming and counting          36/36 boxes, 6/6 correct class multiset
horizontal placement         cx within 0.003 of image width
box size                     systematically 0.65x too small
===========================  =========================================

So it is used as :class:`VisionLLMEvidence` --- an opinion on boxes somebody
else proposed --- wherever a detector already exists. That is the role its
naming accuracy earns, and naming is exactly where the station's detector is
weakest: the whole ``red_orange_critical`` group exists because Red and
Orange are confused.

:class:`VisionLLMProposer` is the cold-start case, where no detector exists
yet and the boxes have to come from somewhere. There the size bias matters,
and it is a *bias* rather than noise --- three constants measured from a
handful of labelled frames corrected held-out frames from 5/18 boxes above
IoU 0.5 to 16/18. Those constants are :class:`BoxCalibration`, and a
proposer without one refuses to propose.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from picture_tool.autotrain import AutoTrainError
from picture_tool.bootstrap.evidence import Box, BoxOpinion, EvidenceError
from picture_tool.bootstrap.profile import ProductProfile
from picture_tool.bootstrap.vision_client import VisionClientError, VisionRequest

#: Name this source answers under, alongside "detector" and "colour".
SOURCE_NAME = "vision_llm"

#: A matched opinion has to be nearer than this, as a fraction of image width.
#: Measured cx error is 0.003 and a box is about 0.045 wide, so 0.02 accepts
#: every real match while refusing to pair a box with an object the model
#: was describing somewhere else entirely.
DEFAULT_MATCH_RADIUS = 0.02

#: Boxes whose reply could not be matched get this, rather than a guess.
UNDECIDED = ""


class CalibrationError(AutoTrainError):
    """Raised when a calibration is missing, malformed, or for another station."""


@dataclass(frozen=True)
class BoxCalibration:
    """The systematic difference between the model's boxes and real ones.

    Three constants, because three is what the measurement supports: the
    model places centres well and draws them too small, consistently. It is
    deliberately not a learned model --- a fitted transform on this little
    data would describe the sample rather than the bias.

    Bound to one station. The framing, lens and working distance are what
    make the bias what it is, so applying a Cable1/A calibration to another
    station would be borrowing a number that was never about it.
    """

    product: str
    area: str
    width_scale: float
    height_scale: float
    cy_shift: float = 0.0
    cx_shift: float = 0.0
    #: How many box pairs the medians were taken over. A caller deciding
    #: whether to trust this needs to know it was six frames, not six hundred.
    sample_boxes: int = 0
    sample_images: int = 0
    derived_from: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.width_scale <= 0 or self.height_scale <= 0:
            raise CalibrationError(
                f"Scales must be positive, got width={self.width_scale}, "
                f"height={self.height_scale}."
            )

    def applies_to(self, product: str, area: str) -> bool:
        return self.product == product and self.area == area

    def apply(self, box: Box) -> Box:
        """One box, corrected. Centres move a little, sizes move a lot."""
        return Box(
            class_name=box.class_name,
            cx=box.cx + self.cx_shift,
            cy=box.cy + self.cy_shift,
            width=box.width * self.width_scale,
            height=box.height * self.height_scale,
            confidence=box.confidence,
            source=box.source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "area": self.area,
            "width_scale": round(self.width_scale, 6),
            "height_scale": round(self.height_scale, 6),
            "cy_shift": round(self.cy_shift, 6),
            "cx_shift": round(self.cx_shift, 6),
            "sample_boxes": self.sample_boxes,
            "sample_images": self.sample_images,
            "derived_from": self.derived_from,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BoxCalibration":
        try:
            return cls(
                product=str(payload["product"]),
                area=str(payload["area"]),
                width_scale=float(payload["width_scale"]),
                height_scale=float(payload["height_scale"]),
                cy_shift=float(payload.get("cy_shift", 0.0)),
                cx_shift=float(payload.get("cx_shift", 0.0)),
                sample_boxes=int(payload.get("sample_boxes", 0)),
                sample_images=int(payload.get("sample_images", 0)),
                derived_from=str(payload.get("derived_from", "")),
                notes=str(payload.get("notes", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"Unreadable calibration: {exc}") from None

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return target

    @classmethod
    def read(cls, path: str | Path) -> "BoxCalibration":
        source = Path(path)
        if not source.is_file():
            raise CalibrationError(f"No calibration at {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise CalibrationError(f"{source} is not readable JSON: {exc}") from None
        return cls.from_dict(payload)


def calibrate(
    pairs: Sequence[tuple[Box, Box]],
    *,
    product: str,
    area: str,
    sample_images: int = 0,
    derived_from: str = "",
    notes: str = "",
    minimum_boxes: int = 6,
) -> BoxCalibration:
    """Measure the bias from boxes the model drew against boxes known good.

    Medians rather than means throughout: one box the model placed on the
    wrong object would drag a mean into a correction that fits nothing, and
    at this sample size there is no room to absorb that.
    """
    if len(pairs) < minimum_boxes:
        raise CalibrationError(
            f"{len(pairs)} box pair(s) is too few to measure a bias; "
            f"{minimum_boxes} is the floor. A correction fitted to fewer "
            "describes those boxes rather than the station."
        )
    widths, heights, dxs, dys = [], [], [], []
    for proposed, reference in pairs:
        if proposed.width <= 0 or proposed.height <= 0:
            raise CalibrationError(
                f"A proposed box has no area ({proposed.width}x{proposed.height}); "
                "it cannot be scaled onto anything."
            )
        widths.append(reference.width / proposed.width)
        heights.append(reference.height / proposed.height)
        dxs.append(reference.cx - proposed.cx)
        dys.append(reference.cy - proposed.cy)
    return BoxCalibration(
        product=product,
        area=area,
        width_scale=statistics.median(widths),
        height_scale=statistics.median(heights),
        cx_shift=statistics.median(dxs),
        cy_shift=statistics.median(dys),
        sample_boxes=len(pairs),
        sample_images=sample_images,
        derived_from=derived_from,
        notes=notes,
    )


def match_by_centre(
    proposed: Sequence[Box],
    targets: Sequence[Box],
    *,
    radius: float = DEFAULT_MATCH_RADIUS,
) -> list[Box | None]:
    """For each target, the nearest proposed box, or None if none is near.

    Nearest-centre rather than best-IoU on purpose. The model's sizes are
    known to be wrong and its centres are known to be right, so IoU would
    rank matches by the one thing that cannot be trusted.
    """
    out: list[Box | None] = []
    for target in targets:
        best: Box | None = None
        best_distance = radius
        for candidate in proposed:
            distance = float(
                np.hypot(candidate.cx - target.cx, candidate.cy - target.cy)
            )
            if distance <= best_distance:
                best, best_distance = candidate, distance
        out.append(best)
    return out


@dataclass(frozen=True)
class VisionPrompt:
    """What the model is told. The station's contract, in words.

    Deliberately says nothing about where objects sit or how big they are.
    An earlier version stated a vertical position, and the resulting boxes
    tracked the stated value rather than the image --- frames whose objects
    sat elsewhere collapsed to IoU 0.03. A prompt that supplies the answer
    cannot be told apart from a model that found it.
    """

    #: What counts as one object. This is the specification, not a hint about
    #: where things are, and leaving it out is not neutrality: asked merely
    #: for "a box around it", the model boxed each whole wire trailing out of
    #: frame instead of the end at its pad, and nothing matched. Saying which
    #: part is the object is the describing half of "send a few pictures and
    #: describe the target". Saying where it sits would be answering for it.
    object_description: str = "For each one, give a tight box around it."

    template: str = (
        "This is a {product} {area} inspection photo. It contains {total} "
        "objects: {inventory}.\n\n"
        "{object_description}\n\n"
        'Reply with ONLY this JSON and nothing else:\n'
        '{{"boxes":[{{"class":"<{classes}>","bbox":[x1,y1,x2,y2]}}]}}\n'
        "with x1,y1,x2,y2 as INTEGERS from 0 to 1000, where 0 is the left or "
        "top edge and 1000 is the right or bottom edge."
    )

    def render(self, profile: ProductProfile) -> str:
        counts = dict(profile.expected_counts)
        inventory = ", ".join(
            f"{count} {name}" for name, count in sorted(counts.items())
        )
        return self.template.format(
            product=profile.product,
            area=profile.area,
            total=sum(counts.values()) or len(profile.class_schema.names),
            inventory=inventory or ", ".join(profile.class_schema.names),
            classes="|".join(profile.class_schema.names),
            object_description=self.object_description,
        )


class VisionLLMEvidence:
    """What a vision model says each box contains.

    One request per image, not per box. The measured strength is naming
    every object in a single look; asking six times would cost six times as
    much to answer the same question with less context. The reply's boxes
    are then matched to the given ones by centre, which the same measurement
    licenses --- centre error is an order of magnitude below box width.

    A box with no reply near it gets an empty opinion rather than a guess.
    Silence is a legitimate answer from an evidence source and is what keeps
    this one independent of the detector it is checking.
    """

    name = SOURCE_NAME

    def __init__(
        self,
        client: Any,
        *,
        prompt: VisionPrompt | None = None,
        match_radius: float = DEFAULT_MATCH_RADIUS,
        max_side: int = 640,
    ) -> None:
        self._client = client
        self._prompt = prompt or VisionPrompt()
        self._match_radius = match_radius
        self._max_side = max_side

    def read(
        self, image: np.ndarray, boxes: Sequence[Box], profile: ProductProfile
    ) -> list[BoxOpinion]:
        if not len(boxes):
            return []
        proposed = self._ask(image, profile)
        matches = match_by_centre(proposed, boxes, radius=self._match_radius)
        opinions = []
        for box, match in zip(boxes, matches):
            if match is None:
                opinions.append(
                    BoxOpinion(
                        source=self.name,
                        class_name=UNDECIDED,
                        confidence=0.0,
                        detail="no object reported near this box",
                    )
                )
                continue
            distance = float(np.hypot(match.cx - box.cx, match.cy - box.cy))
            opinions.append(
                BoxOpinion(
                    source=self.name,
                    class_name=match.class_name,
                    confidence=match.confidence,
                    detail=f"matched at {distance:.4f} of image width",
                )
            )
        return opinions

    def _ask(self, image: np.ndarray, profile: ProductProfile) -> list[Box]:
        import cv2

        height, width = image.shape[:2]
        scale = min(1.0, self._max_side / max(height, width))
        sent = (
            cv2.resize(image, (int(width * scale), int(height * scale)))
            if scale < 1.0
            else image
        )
        ok, buffer = cv2.imencode(".jpg", sent)
        if not ok:
            raise EvidenceError("Could not encode the image for the vision model")
        request = VisionRequest(
            sample_id=f"{profile.product}_{profile.area}",
            prompt=self._prompt.render(profile),
            crops=(bytes(buffer),),
        )
        try:
            # ask(), not judge(): this request wants coordinates, and a reply
            # without a verdict in it is exactly right here rather than an
            # error to be retried twice before giving up.
            reply = self._client.ask(request, require_verdict=False)
        except VisionClientError as exc:
            raise EvidenceError(f"Vision model could not be reached: {exc}") from None
        return boxes_from_reply(reply.fields, profile)


#: The grid the prompt asks for, and the one these models are trained on.
#: Asking in the model's own units is what removes the guesswork: a value of
#: 233 is 0.233 on this grid and 0.364 in a 640-pixel frame, and nothing in
#: the number says which was meant. Replies were seen mixing conventions
#: inside a single box --- x in pixels, y normalised --- so the convention is
#: stated in the prompt rather than inferred from magnitudes here.
COORDINATE_GRID = 1000.0


def _rescale(low: float, high: float) -> tuple[float, float] | None:
    """One axis, from the asked-for grid to 0-1.

    A reply already in 0-1 is accepted too, since that costs nothing and the
    two ranges cannot be confused: a box occupying less than a thousandth of
    the frame is not a wire end. Anything that still falls outside the frame
    is dropped rather than clamped --- a box clamped to the edge is a
    plausible-looking wrong answer, which is the kind this module exists to
    keep out of a dataset.
    """
    scale = 1.0 if high <= 1.0 else COORDINATE_GRID
    low, high = low / scale, high / scale
    if not (0.0 <= low < high <= 1.0):
        return None
    return low, high


def boxes_from_reply(
    payload: Mapping[str, Any], profile: ProductProfile
) -> list[Box]:
    """Normalised boxes out of the model's JSON, dropping what does not fit.

    A class outside the station's contract is dropped rather than renumbered
    or invented, matching what the detector proposer does with the same
    problem: a name the contract does not contain is not a labelling
    decision anybody made.
    """
    raw = payload.get("boxes")
    if not isinstance(raw, list):
        return []
    allowed = set(profile.class_schema.names)
    boxes: list[Box] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("class") or item.get("class_name") or "")
        bbox = item.get("bbox") or item.get("box")
        if name not in allowed or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        xs = _rescale(*sorted((x1, x2)))
        ys = _rescale(*sorted((y1, y2)))
        if xs is None or ys is None:
            continue
        boxes.append(
            Box(
                class_name=name,
                cx=(xs[0] + xs[1]) / 2.0,
                cy=(ys[0] + ys[1]) / 2.0,
                width=xs[1] - xs[0],
                height=ys[1] - ys[0],
                confidence=float(item.get("confidence", 0.0) or 0.0),
                source=SOURCE_NAME,
            )
        )
    return boxes


class VisionLLMProposer:
    """Boxes for a station that has no detector yet.

    Refuses to propose without a :class:`BoxCalibration` for this station.
    Uncorrected boxes measured mean IoU 0.36 against known-good ones, and a
    dataset labelled at that accuracy teaches a detector the wrong extents
    while looking, in every count and class total, entirely correct. The
    failure would surface as a mediocre model much later and with nothing
    pointing back here.
    """

    name = SOURCE_NAME

    def __init__(
        self,
        client: Any,
        *,
        calibration: BoxCalibration | None = None,
        prompt: VisionPrompt | None = None,
        max_side: int = 640,
        allow_uncalibrated: bool = False,
    ) -> None:
        self._evidence = VisionLLMEvidence(
            client, prompt=prompt, max_side=max_side
        )
        self.calibration = calibration
        self.allow_uncalibrated = allow_uncalibrated

    def propose(self, image_path: str | Path, profile: ProductProfile) -> list[Box]:
        import cv2

        calibration = self._checked_calibration(profile)
        image = cv2.imread(str(image_path))
        if image is None:
            raise EvidenceError(f"Could not read {image_path}")
        boxes = self._evidence._ask(image, profile)
        if calibration is None:
            return boxes
        return [calibration.apply(box) for box in boxes]

    def _checked_calibration(self, profile: ProductProfile) -> BoxCalibration | None:
        if self.calibration is None:
            if self.allow_uncalibrated:
                return None
            raise CalibrationError(
                f"No box calibration for {profile.product}/{profile.area}. "
                "Uncorrected boxes measured mean IoU 0.36 against known-good "
                "ones, which is accurate enough to look right and wrong "
                "enough to train on. Derive one with calibrate(), or pass "
                "allow_uncalibrated=True if the boxes are not for training."
            )
        if not self.calibration.applies_to(profile.product, profile.area):
            raise CalibrationError(
                f"Calibration is for {self.calibration.product}/"
                f"{self.calibration.area}, not {profile.product}/{profile.area}. "
                "The bias comes from this station's framing and optics, so it "
                "is not a number another station may borrow."
            )
        return self.calibration


@dataclass(frozen=True)
class AgreementReport:
    """How often two sources named a box the same way."""

    compared: int = 0
    agreed: int = 0
    undecided: int = 0
    disagreements: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)

    @property
    def rate(self) -> float:
        return self.agreed / self.compared if self.compared else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "compared": self.compared,
            "agreed": self.agreed,
            "undecided": self.undecided,
            "agreement_rate": round(self.rate, 4),
            "disagreements": [
                {"box": box, "left": left, "right": right}
                for box, left, right in self.disagreements
            ],
        }


def compare_opinions(
    left: Sequence[BoxOpinion], right: Sequence[BoxOpinion]
) -> AgreementReport:
    """Where two evidence sources differ, which is the interesting part.

    Agreement is not accuracy --- two sources looking at the same pixels can
    be wrong together --- so this reports where they part company and leaves
    the adjudication to whoever has ground truth.
    """
    if len(left) != len(right):
        raise EvidenceError(
            f"Opinion lists must line up with the same boxes: "
            f"{len(left)} against {len(right)}."
        )
    agreed = undecided = 0
    differences: list[tuple[str, str, str]] = []
    for index, (a, b) in enumerate(zip(left, right)):
        if not a.class_name or not b.class_name:
            undecided += 1
            continue
        if a.class_name == b.class_name:
            agreed += 1
        else:
            differences.append((f"box_{index + 1}", a.class_name, b.class_name))
    return AgreementReport(
        compared=len(left) - undecided,
        agreed=agreed,
        undecided=undecided,
        disagreements=tuple(differences),
    )
