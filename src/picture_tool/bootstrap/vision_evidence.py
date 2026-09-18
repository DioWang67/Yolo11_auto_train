"""A vision model as an evidence source, and as a cold-start proposer.

Measured on this station before being written, because the model's strengths
here are lopsided and the architecture follows from which is which. On the
six verified frames of 2026-09-15, asked to name and locate all six wire ends:

===========================  =========================================
naming and counting          36/36 boxes, 6/6 correct class multiset
horizontal placement         cx within 0.003 of image width
box size                     systematically 0.65x too small
vertical placement           flattened; does not follow the board's tilt
===========================  =========================================

The last row is the one that shapes everything below it: cy is the single
coordinate here that is not usable, so nothing in this module may let cy
decide anything cx can decide instead.

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

#: How far apart two centres may be *horizontally* and still be the same
#: object, as a fraction of image width. Measured cx error is 0.003 and a box
#: is about 0.045 wide, so 0.02 accepts every real match while refusing to
#: pair a box with an object the model was describing somewhere else entirely.
DEFAULT_MATCH_RADIUS = 0.02

#: Boxes whose reply could not be matched get this, rather than a guess.
UNDECIDED = ""

#: How many hand-labelled frames a calibration needs under it, and how many
#: are worth asking for. Measured by resampling Cable1/A's 46 frames: the
#: spread in height_scale across draws is 27% of its own value at 3 frames,
#: 18.8% at 5, 12.5% at 8, 10.3% at 12, and 5.2% at 23. Eight is where a
#: held-out frame's IoU stops depending on which frames were picked; twelve
#: buys the last of it. Past twelve the return is 0.014 IoU for more than
#: doubling the labelling, which is not a trade worth asking a person for.
MINIMUM_SEED_IMAGES = 8
RECOMMENDED_SEED_IMAGES = 12

#: Send the image as it was taken. The model's box placement is a function
#: of whether it can resolve the object at all: shrinking a 3072px frame to
#: 640 turns a 125px wire end into 26px, and the measured vertical error
#: goes from 0.0065 of frame height to 0.0771 -- twelve times worse, to save
#: four seconds a frame. Naming is unaffected, which is why
#: :class:`VisionLLMEvidence` still defaults to a thumbnail and only the
#: proposer, whose whole job is placement, defaults to the full frame.
NO_DOWNSCALE = 0


class CalibrationError(AutoTrainError):
    """Raised when a calibration is missing, malformed, or for another station."""


@dataclass(frozen=True)
class BoxCalibration:
    """The systematic difference between the model's boxes and real ones.

    A few constants, because constants are what the measurement supports:
    the model places centres well and draws them too small, consistently.
    It is deliberately not a learned model --- a fitted transform on this
    little data would describe the sample rather than the bias.

    Bound to one station, **one image scale and one model**. The framing,
    lens and working distance are what make the bias what it is, so applying
    a Cable1/A calibration to another station would be borrowing a number
    that was never about it --- and the same is true of the other two.

    The scale is not a footnote. Measured on the same station, same frames,
    same prompt: at ``max_side`` 640 the height correction is x1.497, at
    3072 it is **x1.981**. Shrinking the image shrinks a 125px wire end to
    26px, and a model that cannot resolve the object draws a vaguer, larger
    box than one that can. A calibration used at a scale it was not derived
    at is wrong by a third and nothing about the boxes looks unusual, so
    :meth:`applies_to` refuses the mismatch rather than trusting the caller
    to remember.

    **``cy_shift`` is the weak term and no tilt term joins it.** Cable1/A's
    46 hand-labelled frames were measured before this was written: within a
    frame the wire ends are collinear to 0.005 of frame height, so a frame's
    vertical layout is two numbers. But across frames the line's slope has a
    spread of 0.117 around a median of 0.067 and reverses sign in 16 of the
    46 --- the board is placed at a different angle each time. A slope like
    that is not a property of the station, so it cannot be carried by a
    class that is `applies_to` one station; a constant fitted to it would be
    the median of a distribution straddling zero. :attr:`cy_shift_spread`
    exists so a caller can see that for itself rather than read
    ``cy_shift`` as though it were as solid as the scales beside it.
    """

    product: str
    area: str
    width_scale: float
    height_scale: float
    #: The longest side the images were resized to before being asked about,
    #: and the model that answered. Both are part of what the bias *is*, so
    #: both are part of what it applies to. Zero and empty mean a calibration
    #: written before this was recorded: readable, but not usable, because
    #: there is no way to tell which scale produced it.
    max_side: int = 0
    model: str = ""
    cy_shift: float = 0.0
    cx_shift: float = 0.0
    #: Median absolute deviation of the vertical corrections the fit saw.
    #: A shift smaller than its own spread is a number the data did not
    #: support --- see the class docstring for why that is expected here
    #: and what has to supply the vertical instead.
    cy_shift_spread: float = 0.0
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

    def mismatch_for(
        self, product: str, area: str, *, max_side: int, model: str
    ) -> str:
        """Why this calibration does not apply here, or "" if it does.

        A reason rather than a bool, because all four ways of not applying
        need different things done about them and a caller holding only
        False has to guess which it hit.

        An unrecorded scale or model is a mismatch, not a pass. The same
        rule already governs vision review, where a result without a model
        field is treated exactly like one from a different model: the
        records that predate the field are precisely the ones the check
        exists to catch, so letting them through waives it where it is most
        needed.
        """
        if self.product != product or self.area != area:
            return (
                f"derived for {self.product}/{self.area}, not {product}/{area}. "
                "The bias comes from this station's framing and optics, so it "
                "is not a number another station may borrow."
            )
        if not self.max_side or not self.model:
            return (
                "was written before the image scale and model were recorded, "
                "so there is no way to tell which scale produced it. Derive it "
                "again; the same measurement at 640 and at 3072 differs by a "
                "third in height_scale."
            )
        if self.max_side != max_side:
            return (
                f"was derived at max_side {self.max_side} and is being used at "
                f"{max_side}. Box size bias tracks how well the model could "
                "resolve the object, so the two are not interchangeable."
            )
        if self.model != model:
            return (
                f"was derived from {self.model}, not {model}. How large a box "
                "a model draws around the same object is the model's own habit."
            )
        return ""

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
            "max_side": self.max_side,
            "model": self.model,
            "cy_shift": round(self.cy_shift, 6),
            "cy_shift_spread": round(self.cy_shift_spread, 6),
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
                # Absent in files written before these were recorded. They
                # read back cleanly and mismatch_for then refuses them, so
                # the failure is a message about re-deriving rather than a
                # crash on load.
                max_side=int(payload.get("max_side", 0) or 0),
                model=str(payload.get("model", "")),
                cy_shift=float(payload.get("cy_shift", 0.0)),
                cy_shift_spread=float(payload.get("cy_shift_spread", 0.0)),
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
    max_side: int = 0,
    model: str = "",
    derived_from: str = "",
    notes: str = "",
    minimum_boxes: int = 6,
    minimum_images: int = MINIMUM_SEED_IMAGES,
) -> BoxCalibration:
    """Measure the bias from boxes the model drew against boxes known good.

    Medians rather than means throughout: one box the model placed on the
    wrong object would drag a mean into a correction that fits nothing, and
    at this sample size there is no room to absorb that.

    The floor that matters is **images, not boxes**. Measured by resampling
    Cable1/A's 46 hand-labelled frames: the spread in ``height_scale`` falls
    with the number of frames, not with the number of boxes in them, because
    what varies between one board and the next --- where it sits, how it
    leans --- is a property of the frame that all six of its boxes share.
    Six boxes from one frame is one observation of the bias wearing the
    disguise of six.

    The vertical correction is reported with its spread beside it, because
    on this station it is expected to be a number without a bias underneath
    it. See :class:`BoxCalibration` for the measurement that says so.
    """
    if len(pairs) < minimum_boxes:
        raise CalibrationError(
            f"{len(pairs)} box pair(s) is too few to measure a bias; "
            f"{minimum_boxes} is the floor. A correction fitted to fewer "
            "describes those boxes rather than the station."
        )
    if not sample_images and minimum_images:
        raise CalibrationError(
            "The number of labelled frames was not recorded, so there is no "
            "way to tell whether the bias was measured over enough of them. "
            "An unrecorded count is refused rather than waved through: "
            "BoxCalibration.mismatch_for applies the same rule to an "
            "unrecorded scale or model, and for the same reason --- the "
            "calls that predate the field are precisely the ones the floor "
            "exists to catch. Pass sample_images, or minimum_images=0 to say "
            "the floor does not apply to this call."
        )
    if sample_images < minimum_images:
        raise CalibrationError(
            f"{sample_images} labelled image(s) is too few; {minimum_images} "
            "is the floor. Resampling this station's hand labels put the "
            "height correction's spread at 27% of its own value over 3 "
            f"images, 12.5% over {MINIMUM_SEED_IMAGES} and 10% over "
            f"{RECOMMENDED_SEED_IMAGES} -- below the floor the correction is "
            "as likely to be 1.07 as 1.88. Label more frames rather than "
            "more boxes in the same frames."
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
    cy_shift = statistics.median(dys)
    return BoxCalibration(
        product=product,
        area=area,
        width_scale=statistics.median(widths),
        height_scale=statistics.median(heights),
        cx_shift=statistics.median(dxs),
        cy_shift=cy_shift,
        cy_shift_spread=statistics.median([abs(dy - cy_shift) for dy in dys]),
        max_side=max_side,
        model=model,
        sample_boxes=len(pairs),
        sample_images=sample_images,
        derived_from=derived_from,
        notes=notes,
    )


#: Two boxes nearer than this in cx say nothing about a slope; the division
#: would amplify label noise instead of measuring a gradient.
MIN_CX_SEPARATION = 0.02

#: Below this many boxes, a frame's own line is not worth fitting.
MIN_BOXES_FOR_A_FRAME_LINE = 4


def _theil_sen_slope(points: Sequence[tuple[float, float]]) -> float | None:
    """Median of the pairwise slopes, or None if x barely moves.

    Least squares would let one box the model put on the wrong object set
    the slope, which is the same reason :func:`calibrate` uses medians.
    """
    slopes = [
        (y2 - y1) / (x2 - x1)
        for i, (x1, y1) in enumerate(points)
        for x2, y2 in points[i + 1:]
        if abs(x2 - x1) >= MIN_CX_SEPARATION
    ]
    return statistics.median(slopes) if slopes else None


@dataclass(frozen=True)
class Spread:
    """A robust summary of one set of measurements.

    ``negative`` is here because a correction whose sign is not settled is
    a different kind of useless from one that is merely imprecise, and the
    median alone hides the difference.
    """

    n: int = 0
    median: float = 0.0
    mad: float = 0.0
    low: float = 0.0
    high: float = 0.0
    negative: int = 0

    @classmethod
    def of(cls, values: Sequence[float]) -> "Spread":
        if not values:
            return cls()
        centre = statistics.median(values)
        return cls(
            n=len(values),
            median=centre,
            mad=statistics.median([abs(v - centre) for v in values]),
            low=min(values),
            high=max(values),
            negative=sum(1 for v in values if v < 0),
        )

    @property
    def settled(self) -> bool:
        """Whether these agree well enough to stand in for one another.

        Says nothing about whether the agreed value is large enough to be
        worth acting on --- measurements that agree on zero are settled and
        not worth storing, and conflating the two reported a perfectly
        consistent absence of tilt as frames disagreeing about one.
        """
        return (
            self.n > 0
            and self.negative in (0, self.n)
            and self.mad < abs(self.median)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "median": round(self.median, 4),
            "mad": round(self.mad, 4),
            "min": round(self.low, 4),
            "max": round(self.high, 4),
            "negative": self.negative,
        }


@dataclass(frozen=True)
class VerticalStructure:
    """Where the vertical error lives: in the station, or in each frame.

    :class:`BoxCalibration` can only carry the first kind. This measures
    which kind is actually present, so that the decision not to add a
    vertical term to it is a reading of data rather than a preference.

    ``between_frames``
        How far frames' own median corrections sit from one another. A
        single ``cy_shift`` can only be right for one of them, so this
        spread is what that shift leaves behind.
    ``within_frames``
        What remains inside a frame once its own median is removed --- the
        part a per-frame offset would still miss, and what a tilt would
        have to explain.
    ``frame_slopes``
        That leftover's gradient against cx, per frame. A station tilt term
        is only meaningful if these agree; if the spread rivals the median
        or the sign flips between frames, there is no station tilt to store.
    """

    between_frames: Spread = field(default_factory=Spread)
    within_frames: Spread = field(default_factory=Spread)
    frame_slopes: Spread = field(default_factory=Spread)

    #: A per-frame offset is only worth having if frames differ by more
    #: than this. Cable1/A's boxes are 0.09 high, so 0.01 is a ninth of a
    #: box --- below it, the offset is not what is hurting IoU.
    MEANINGFUL_OFFSET = 0.01

    #: A slope moves cy by ``slope * cx-span`` across a frame. Cable1/A's
    #: span is 0.26 and its boxes are 0.09 high, so 0.05 is where the tilt
    #: starts to move a box by a seventh of itself. Below that there is
    #: nothing to correct, however consistently the frames agree on it.
    MEANINGFUL_TILT = 0.05

    @property
    def reading(self) -> str:
        """The conclusion, so a reader need not re-derive it from numbers.

        Only ever reports what is *wrong*; a run with nothing to report
        falls through to saying a constant fits. So a flat tilt that every
        frame agrees on is silence here, not a complaint about tilt.
        """
        if not self.between_frames.n:
            return "No matched boxes; nothing measured."
        notes = []
        if self.between_frames.mad > self.MEANINGFUL_OFFSET:
            notes.append(
                f"frames disagree about the vertical by "
                f"{self.between_frames.mad:.4f} (MAD), so no one "
                "station-wide cy_shift can serve them all"
            )
        slopes = self.frame_slopes
        tilted = abs(slopes.median) >= self.MEANINGFUL_TILT
        # Spread counts on its own: frames leaning hard in both directions
        # average out to a median that looks flat and is not.
        if slopes.n and (tilted or slopes.mad >= self.MEANINGFUL_TILT):
            if slopes.settled:
                notes.append(
                    f"per-frame tilt agrees at {slopes.median:+.4f} "
                    f"+/-{slopes.mad:.4f}; a station tilt term would earn "
                    "its place"
                )
            else:
                notes.append(
                    f"per-frame tilt is {slopes.median:+.4f} "
                    f"+/-{slopes.mad:.4f} with {slopes.negative}/{slopes.n} "
                    "leaning the other way, so there is no station tilt "
                    "to store either"
                )
        return "; ".join(notes) or "A constant vertical shift fits this data."

    def to_dict(self) -> dict[str, Any]:
        return {
            "between_frames": self.between_frames.to_dict(),
            "within_frames": self.within_frames.to_dict(),
            "frame_slopes": self.frame_slopes.to_dict(),
            "reading": self.reading,
        }


def measure_vertical_structure(
    per_frame: Sequence[Sequence[tuple[Box, Box]]]
) -> VerticalStructure:
    """Split the vertical error into what a constant could fix and what not.

    Takes pairs still grouped by the frame they came from, because that
    grouping *is* the measurement: flattened, between-frame and
    within-frame error are indistinguishable, and the question of which one
    dominates is the question being asked.
    """
    frame_medians: list[float] = []
    within: list[float] = []
    slopes: list[float] = []
    for pairs in per_frame:
        residuals = [
            (proposed.cx, reference.cy - proposed.cy)
            for proposed, reference in pairs
        ]
        if not residuals:
            continue
        centre = statistics.median([dy for _, dy in residuals])
        frame_medians.append(centre)
        within.extend(dy - centre for _, dy in residuals)
        if len(residuals) >= MIN_BOXES_FOR_A_FRAME_LINE:
            slope = _theil_sen_slope(residuals)
            if slope is not None:
                slopes.append(slope)
    return VerticalStructure(
        between_frames=Spread.of(frame_medians),
        within_frames=Spread.of(within),
        frame_slopes=Spread.of(slopes),
    )


def match_by_centre(
    proposed: Sequence[Box],
    targets: Sequence[Box],
    *,
    radius: float = DEFAULT_MATCH_RADIUS,
    vertical_radius: float | None = None,
) -> list[Box | None]:
    """For each target, the nearest proposed box, or None if none is near.

    Nearest-centre rather than best-IoU on purpose. The model's sizes are
    known to be wrong and its centres are known to be right, so IoU would
    rank matches by the one thing that cannot be trusted.

    **Horizontal distance alone decides the match.** An earlier version took
    the plain two-dimensional distance, which spent the model's one accurate
    coordinate on its one inaccurate one: frames whose six boxes were all
    within 0.005 of the right cx and named 6/6 correctly came back 0/6
    matched, because cy can be out by 0.06 and the radius is 0.02. Vetoing a
    trustworthy axis with an untrustworthy one is not a strict test, it is a
    broken one.

    ``vertical_radius`` re-admits cy as a *gate* --- never as a ranking term
    --- for a station whose objects sit above one another and so cannot be
    told apart by cx. Cable1/A's are a single row, 0.045 apart horizontally,
    so it is left off there and cy is only ever a tie-break.
    """
    if radius <= 0:
        raise EvidenceError(f"A match radius must be positive, got {radius}.")
    if vertical_radius is not None and vertical_radius <= 0:
        raise EvidenceError(
            f"A vertical match radius must be positive, got {vertical_radius}."
        )
    out: list[Box | None] = []
    for target in targets:
        best: Box | None = None
        best_key: tuple[float, float] | None = None
        for candidate in proposed:
            dx = abs(candidate.cx - target.cx)
            if dx > radius:
                continue
            dy = abs(candidate.cy - target.cy)
            if vertical_radius is not None and dy > vertical_radius:
                continue
            # dx first, so cy can only separate candidates cx cannot.
            key = (dx, dy)
            if best_key is None or key < best_key:
                best, best_key = candidate, key
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
        vertical_match_radius: float | None = None,
        max_side: int = 640,
    ) -> None:
        self._client = client
        self._prompt = prompt or VisionPrompt()
        self._match_radius = match_radius
        self._vertical_match_radius = vertical_match_radius
        self._max_side = max_side

    def read(
        self, image: np.ndarray, boxes: Sequence[Box], profile: ProductProfile
    ) -> list[BoxOpinion]:
        if not len(boxes):
            return []
        proposed = self._ask(image, profile)
        matches = match_by_centre(
            proposed,
            boxes,
            radius=self._match_radius,
            vertical_radius=self._vertical_match_radius,
        )
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
            # Horizontal only, because that is what decided the match. A
            # combined distance here would report a number no rule used.
            offset = abs(match.cx - box.cx)
            opinions.append(
                BoxOpinion(
                    source=self.name,
                    class_name=match.class_name,
                    confidence=match.confidence,
                    detail=f"matched {offset:.4f} of image width away in cx",
                )
            )
        return opinions

    def _ask(self, image: np.ndarray, profile: ProductProfile) -> list[Box]:
        import cv2

        height, width = image.shape[:2]
        # A max_side of zero means send it as taken; without this the scale
        # below is 0.0 and cv2 is asked to resize to nothing.
        scale = (
            min(1.0, self._max_side / max(height, width))
            if self._max_side > 0
            else 1.0
        )
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
        max_side: int = NO_DOWNSCALE,
        model: str = "",
        allow_uncalibrated: bool = False,
    ) -> None:
        self._evidence = VisionLLMEvidence(
            client, prompt=prompt, max_side=max_side
        )
        self.max_side = max_side
        self.model = model
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
        reason = self.calibration.mismatch_for(
            profile.product, profile.area,
            max_side=self.max_side, model=self.model,
        )
        if reason:
            raise CalibrationError(f"This calibration {reason}")
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
