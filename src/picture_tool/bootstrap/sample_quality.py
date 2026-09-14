"""Is this image worth trusting at all --- said without looking at classes.

Deliberately narrow. Quality answers one question: could a person or a model
read this frame reliably. It never says what is in it. Mixing the two is how
"the image is dark" turns into "the wire is Black", and a pseudo-label built
on that reasoning is wrong in a way that looks confident.

Thresholds are far out in the tails on purpose. An ordinary dim frame is
normal for this line and must not be thrown away --- a dataset filtered to
well-lit images teaches a model that the line is always well lit. Only the
failures that make a frame genuinely unreadable are refused here; anything
merely unusual is reported and left for the decision layer to weigh
alongside everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from picture_tool.autotrain.golden_candidates import difference_hash, sha256_file
from picture_tool.autotrain.image_quality import measure_file

#: The frame is readable.
QUALITY_PASS = "PASS"
#: Readable, but something about it should count against acceptance.
QUALITY_SUSPECT = "SUSPECT"
#: Not readable; no label drawn on it can be trusted.
QUALITY_FAIL = "FAIL"

REASON_UNREADABLE = "unreadable_image"
REASON_ABNORMAL_SIZE = "abnormal_dimensions"
REASON_SEVERE_BLUR = "severe_blur"
REASON_SEVERE_OVEREXPOSURE = "severe_overexposure"
REASON_SEVERE_UNDEREXPOSURE = "severe_underexposure"
REASON_EXACT_DUPLICATE = "exact_duplicate"
REASON_NEAR_DUPLICATE = "near_duplicate"
REASON_LOW_SATURATION = "low_saturation"


@dataclass(frozen=True)
class QualityThresholds:
    """Where "unusual" ends and "unreadable" begins.

    The blur floor is a variance-of-Laplacian, and 15 is roughly an order of
    magnitude below this station's median of 66 --- a frame that far down is
    not a slightly soft picture, it is smeared. The exposure bounds are near
    the ends of the 8-bit range for the same reason: a mean of 8 or 248 means
    detail is gone, not that the lamp drifted.
    """

    min_blur: float = 15.0
    min_brightness: float = 12.0
    max_brightness: float = 245.0
    #: Only a diagnostic; a washed-out frame is suspect, never refused, since
    #: saturation is exactly what a colour fault would lower.
    low_saturation: float = 8.0
    min_pixels: int = 64


@dataclass(frozen=True)
class SampleQuality:
    """One image's readability, with the numbers behind the verdict."""

    sample_id: str
    image_path: str
    status: str
    reasons: tuple[str, ...] = ()
    detail: str = ""
    sha256: str = ""
    perceptual_hash: str = ""
    width: int = 0
    height: int = 0
    brightness: float | None = None
    saturation: float | None = None
    blur_score: float | None = None
    duplicate_of: str = ""

    @property
    def is_usable(self) -> bool:
        return self.status != QUALITY_FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "status": self.status,
            "reasons": list(self.reasons),
            "detail": self.detail,
            "sha256": self.sha256,
            "perceptual_hash": self.perceptual_hash,
            "width": self.width,
            "height": self.height,
            "brightness": self.brightness,
            "saturation": self.saturation,
            "blur_score": self.blur_score,
            "duplicate_of": self.duplicate_of,
        }


def assess(
    image_path: str | Path,
    *,
    sample_id: str = "",
    thresholds: QualityThresholds | None = None,
    seen_sha256: Mapping[str, str] | None = None,
    seen_hashes: Mapping[str, str] | None = None,
) -> SampleQuality:
    """Measure one image. Reads it; never writes, never moves it.

    ``seen_sha256`` and ``seen_hashes`` map an already-accepted identity to
    the sample that claimed it, so duplicates are found against what the
    batch has taken so far rather than against the whole world. Exact
    duplicates fail --- two copies of one file are one sample, and the second
    only adds weight to whatever the first got wrong. Near duplicates are
    marked suspect instead, because at this station two genuinely different
    photographs of one fixture do sometimes hash alike.
    """
    path = Path(image_path)
    limits = thresholds or QualityThresholds()
    identifier = sample_id or path.stem

    measured = measure_file(path)
    if measured is None:
        return SampleQuality(
            sample_id=identifier,
            image_path=str(path),
            status=QUALITY_FAIL,
            reasons=(REASON_UNREADABLE,),
            detail="the image could not be decoded",
        )

    digest = ""
    try:
        digest = sha256_file(path)
    except OSError:
        return SampleQuality(
            sample_id=identifier,
            image_path=str(path),
            status=QUALITY_FAIL,
            reasons=(REASON_UNREADABLE,),
            detail="the file could not be read",
        )
    perceptual = difference_hash(path) or ""

    width, height = _dimensions(path)
    reasons: list[str] = []
    details: list[str] = []
    duplicate_of = ""

    if width < limits.min_pixels or height < limits.min_pixels:
        reasons.append(REASON_ABNORMAL_SIZE)
        details.append(f"{width}x{height} is too small to contain the fixture")

    if measured.blur_score < limits.min_blur:
        reasons.append(REASON_SEVERE_BLUR)
        details.append(f"blur score {measured.blur_score:.1f} < {limits.min_blur}")
    if measured.brightness > limits.max_brightness:
        reasons.append(REASON_SEVERE_OVEREXPOSURE)
        details.append(f"brightness {measured.brightness:.1f} is clipped")
    elif measured.brightness < limits.min_brightness:
        reasons.append(REASON_SEVERE_UNDEREXPOSURE)
        details.append(f"brightness {measured.brightness:.1f} is near black")

    if digest and digest in (seen_sha256 or {}):
        duplicate_of = (seen_sha256 or {})[digest]
        reasons.append(REASON_EXACT_DUPLICATE)
        details.append(f"byte-identical to {duplicate_of}")

    suspect: list[str] = []
    if perceptual and perceptual in (seen_hashes or {}):
        duplicate_of = duplicate_of or (seen_hashes or {})[perceptual]
        suspect.append(REASON_NEAR_DUPLICATE)
        details.append(f"looks like {duplicate_of}")
    if measured.saturation < limits.low_saturation:
        suspect.append(REASON_LOW_SATURATION)
        details.append(f"saturation {measured.saturation:.1f} is very low")

    if reasons:
        status = QUALITY_FAIL
    elif suspect:
        status = QUALITY_SUSPECT
    else:
        status = QUALITY_PASS

    return SampleQuality(
        sample_id=identifier,
        image_path=str(path),
        status=status,
        reasons=tuple(reasons + suspect),
        detail="; ".join(details),
        sha256=digest,
        perceptual_hash=perceptual,
        width=width,
        height=height,
        brightness=measured.brightness,
        saturation=measured.saturation,
        blur_score=measured.blur_score,
        duplicate_of=duplicate_of,
    )


def assess_batch(
    image_paths: Iterable[str | Path],
    *,
    thresholds: QualityThresholds | None = None,
    sample_ids: Mapping[str, str] | None = None,
) -> list[SampleQuality]:
    """Assess a batch, accumulating duplicate identities as it goes.

    Order matters and is the caller's to choose: the first occurrence of an
    identity is the one kept, so a caller that wants the best copy first
    should sort before calling rather than hope.
    """
    seen_sha: dict[str, str] = {}
    seen_hash: dict[str, str] = {}
    results: list[SampleQuality] = []
    for path in image_paths:
        identifier = (sample_ids or {}).get(str(path), "") or Path(path).stem
        result = assess(
            path,
            sample_id=identifier,
            thresholds=thresholds,
            seen_sha256=seen_sha,
            seen_hashes=seen_hash,
        )
        if result.sha256 and result.sha256 not in seen_sha:
            seen_sha[result.sha256] = result.sample_id
        if result.perceptual_hash and result.perceptual_hash not in seen_hash:
            seen_hash[result.perceptual_hash] = result.sample_id
        results.append(result)
    return results


def _dimensions(path: Path) -> tuple[int, int]:
    try:
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            return (0, 0)
        return (int(image.shape[1]), int(image.shape[0]))
    except Exception:  # noqa: BLE001 - opencv raises broadly on odd files
        return (0, 0)


DEFAULT_THRESHOLDS = QualityThresholds()

__all__ = [
    "DEFAULT_THRESHOLDS",
    "QUALITY_FAIL",
    "QUALITY_PASS",
    "QUALITY_SUSPECT",
    "QualityThresholds",
    "SampleQuality",
    "assess",
    "assess_batch",
]

_REASONS: tuple[str, ...] = (
    REASON_UNREADABLE,
    REASON_ABNORMAL_SIZE,
    REASON_SEVERE_BLUR,
    REASON_SEVERE_OVEREXPOSURE,
    REASON_SEVERE_UNDEREXPOSURE,
    REASON_EXACT_DUPLICATE,
    REASON_NEAR_DUPLICATE,
    REASON_LOW_SATURATION,
)
