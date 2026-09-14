"""Configuration and the feature flag for autonomous training.

The whole subsystem is gated on ``autonomous_training.enabled``. Every entry
point calls :meth:`AutoTrainConfig.require_enabled` before touching the
filesystem, so a disabled installation behaves exactly as it did before this
package existed.

Thresholds are all configurable and all validated here, at the boundary, so
downstream code can treat an :class:`AutoTrainConfig` as already sane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from picture_tool.autotrain import AutoTrainDisabledError, AutoTrainError

#: Floor for a per-group golden score. Below this many samples the evaluator
#: reports INSUFFICIENT instead of a number: a rate over a handful of images
#: moves in whole-sample steps while still reading like a measurement.
#: Lives here rather than in the evaluator so that the default and the
#: configurable setting cannot drift apart.
DEFAULT_MIN_GROUP_SAMPLES = 10

#: Location of the settings file inside the training project.
DEFAULT_CONFIG_RELPATH = Path("configs") / "autonomous_training.yaml"

#: Top-level key, kept explicit so the file can gain unrelated sections later.
ROOT_KEY = "autonomous_training"

#: Selector names this phase implements. Names outside this set are rejected
#: rather than ignored: a typo that silently disables a selector would show up
#: as "the pool is smaller than expected" long after the run.
IMPLEMENTED_SELECTORS = ("low_confidence", "review_correction", "random_sample")

#: Registered but deliberately unimplemented; accepted in config so the shape
#: is stable, refused if switched on.
PLANNED_SELECTORS = (
    "model_disagreement",
    "class_imbalance",
    "embedding_novelty",
    "distribution_drift",
)


class AutoTrainConfigError(AutoTrainError):
    """Raised when the settings file is missing, malformed or out of range."""


@dataclass(frozen=True)
class StationConfig:
    """The product/area this configuration is written for."""

    product: str
    area: str


@dataclass(frozen=True)
class CollectorConfig:
    """How much production history one collection pass reads."""

    lookback_days: int = 7
    max_records: int = 5000
    compute_image_quality: bool = True


@dataclass(frozen=True)
class SelectorSettings:
    """One selector's switch plus its own options."""

    name: str
    enabled: bool
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldenConfig:
    """Pointer to the locked evaluation dataset.

    Empty ``dataset_path`` means "not configured", which is a first-class
    state: evaluation reports ``NOT_CONFIGURED`` and promotion is refused.
    """

    dataset_path: str = ""
    manifest_sha256: str = ""
    #: Floor for a per-group score; see :data:`DEFAULT_MIN_GROUP_SAMPLES`.
    min_group_samples: int = DEFAULT_MIN_GROUP_SAMPLES

    @property
    def is_configured(self) -> bool:
        return bool(self.dataset_path.strip())


@dataclass(frozen=True)
class TrainingConfig:
    """Challenger training parameters."""

    base_model: str = "champion"
    epochs: int = 50
    imgsz: int = 640
    device: str = "auto"
    batch: int = 4


@dataclass(frozen=True)
class PromotionConfig:
    """Deterministic promotion thresholds.

    ``max_overall_regression`` mirrors the existing deployment gate's own
    0.02 default so the two gates do not disagree by accident.
    """

    min_map50_delta: float = 0.0
    max_overall_regression: float = 0.02
    critical_classes: tuple[str, ...] = ()
    max_critical_class_recall_drop: float = 0.005
    max_false_negatives: int = 0
    require_golden_pass: bool = True


@dataclass(frozen=True)
class AutoTrainConfig:
    """Validated settings for one station's autonomous training path."""

    enabled: bool
    station: StationConfig
    collector: CollectorConfig
    selectors: tuple[SelectorSettings, ...]
    golden: GoldenConfig
    training: TrainingConfig
    promotion: PromotionConfig
    source_path: Path | None = None

    # -- construction -----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AutoTrainConfig":
        """Read and validate the settings file.

        A missing file is not an error: it means the feature was never set up,
        which is indistinguishable in effect from being disabled.
        """
        resolved = cls._resolve_path(path)
        if resolved is None or not resolved.is_file():
            return cls.disabled(source_path=resolved)
        try:
            payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            raise AutoTrainConfigError(
                f"Unable to read autonomous training config {resolved}: {exc}"
            ) from exc
        if payload is None:
            return cls.disabled(source_path=resolved)
        if not isinstance(payload, dict):
            raise AutoTrainConfigError(
                f"{resolved} must contain a YAML mapping at the top level."
            )
        section = payload.get(ROOT_KEY)
        if section is None:
            return cls.disabled(source_path=resolved)
        if not isinstance(section, dict):
            raise AutoTrainConfigError(
                f"{resolved}: {ROOT_KEY!r} must be a mapping."
            )
        return cls.from_mapping(section, source_path=resolved)

    @classmethod
    def disabled(cls, *, source_path: Path | None = None) -> "AutoTrainConfig":
        """Return an inert configuration that refuses every entry point."""
        return cls(
            enabled=False,
            station=StationConfig(product="", area=""),
            collector=CollectorConfig(),
            selectors=(),
            golden=GoldenConfig(),
            training=TrainingConfig(),
            promotion=PromotionConfig(),
            source_path=source_path,
        )

    @classmethod
    def from_mapping(
        cls, section: Mapping[str, Any], *, source_path: Path | None = None
    ) -> "AutoTrainConfig":
        """Build a validated configuration from the parsed mapping."""
        enabled = _as_bool(section.get("enabled", False), "enabled")
        station_raw = _as_mapping(section.get("station"), "station")
        collector_raw = _as_mapping(section.get("collector"), "collector")
        selectors_raw = _as_mapping(section.get("selectors"), "selectors")
        golden_raw = _as_mapping(section.get("golden"), "golden")
        training_raw = _as_mapping(section.get("training"), "training")
        promotion_raw = _as_mapping(section.get("promotion"), "promotion")

        config = cls(
            enabled=enabled,
            station=StationConfig(
                product=str(station_raw.get("product", "")).strip(),
                area=str(station_raw.get("area", "")).strip(),
            ),
            collector=CollectorConfig(
                lookback_days=_as_int(
                    collector_raw.get("lookback_days", 7),
                    "collector.lookback_days",
                    minimum=1,
                ),
                max_records=_as_int(
                    collector_raw.get("max_records", 5000),
                    "collector.max_records",
                    minimum=1,
                ),
                compute_image_quality=_as_bool(
                    collector_raw.get("compute_image_quality", True),
                    "collector.compute_image_quality",
                ),
            ),
            selectors=_parse_selectors(selectors_raw),
            golden=GoldenConfig(
                dataset_path=str(golden_raw.get("dataset_path", "") or "").strip(),
                manifest_sha256=str(
                    golden_raw.get("manifest_sha256", "") or ""
                ).strip(),
                min_group_samples=_as_int(
                    golden_raw.get(
                        "min_group_samples", DEFAULT_MIN_GROUP_SAMPLES
                    ),
                    "golden.min_group_samples",
                    minimum=1,
                ),
            ),
            training=TrainingConfig(
                base_model=str(training_raw.get("base_model", "champion")).strip()
                or "champion",
                epochs=_as_int(
                    training_raw.get("epochs", 50), "training.epochs", minimum=1
                ),
                imgsz=_as_int(
                    training_raw.get("imgsz", 640), "training.imgsz", minimum=32
                ),
                device=str(training_raw.get("device", "auto")).strip() or "auto",
                batch=_as_int(
                    training_raw.get("batch", 4), "training.batch", minimum=1
                ),
            ),
            promotion=PromotionConfig(
                min_map50_delta=_as_float(
                    promotion_raw.get("min_map50_delta", 0.0),
                    "promotion.min_map50_delta",
                ),
                max_overall_regression=_as_float(
                    promotion_raw.get("max_overall_regression", 0.02),
                    "promotion.max_overall_regression",
                    minimum=0.0,
                ),
                critical_classes=_as_str_tuple(
                    promotion_raw.get("critical_classes", ()),
                    "promotion.critical_classes",
                ),
                max_critical_class_recall_drop=_as_float(
                    promotion_raw.get("max_critical_class_recall_drop", 0.005),
                    "promotion.max_critical_class_recall_drop",
                    minimum=0.0,
                ),
                max_false_negatives=_as_int(
                    promotion_raw.get("max_false_negatives", 0),
                    "promotion.max_false_negatives",
                    minimum=0,
                ),
                require_golden_pass=_as_bool(
                    promotion_raw.get("require_golden_pass", True),
                    "promotion.require_golden_pass",
                ),
            ),
            source_path=source_path,
        )
        if config.enabled:
            config._validate_enabled()
        return config

    # -- use --------------------------------------------------------------------

    def require_enabled(self) -> None:
        """Refuse to proceed unless the feature flag is on.

        Called first by every entry point, before any directory is created.
        """
        if not self.enabled:
            location = self.source_path or DEFAULT_CONFIG_RELPATH
            raise AutoTrainDisabledError(
                "Autonomous training is disabled. Set "
                f"{ROOT_KEY}.enabled: true in {location} to enable it."
            )

    def selector(self, name: str) -> SelectorSettings | None:
        """Return one selector's settings, or None when it is not configured."""
        for selector in self.selectors:
            if selector.name == name:
                return selector
        return None

    def enabled_selectors(self) -> tuple[SelectorSettings, ...]:
        """Return the configured selectors that are switched on."""
        return tuple(selector for selector in self.selectors if selector.enabled)

    # -- internals ---------------------------------------------------------------

    def _validate_enabled(self) -> None:
        """Checks that only matter once the feature is actually switched on."""
        if not self.station.product or not self.station.area:
            raise AutoTrainConfigError(
                "station.product and station.area are required when "
                f"{ROOT_KEY}.enabled is true."
            )
        for part, label in (
            (self.station.product, "station.product"),
            (self.station.area, "station.area"),
        ):
            if part in {".", ".."} or "/" in part or "\\" in part:
                raise AutoTrainConfigError(
                    f"{label} must be a single path segment: {part!r}"
                )
        if not self.enabled_selectors():
            raise AutoTrainConfigError(
                "At least one selector must be enabled when "
                f"{ROOT_KEY}.enabled is true."
            )

    @staticmethod
    def _resolve_path(path: str | Path | None) -> Path | None:
        if path is not None:
            return Path(path).expanduser().resolve()
        project_root = Path(__file__).resolve().parents[3]
        return (project_root / DEFAULT_CONFIG_RELPATH).resolve()


# ---------------------------------------------------------------------------
# Coercion helpers. Each names the offending key, because a config error the
# operator cannot locate is barely better than a silent default.


def _parse_selectors(raw: Mapping[str, Any]) -> tuple[SelectorSettings, ...]:
    known = set(IMPLEMENTED_SELECTORS) | set(PLANNED_SELECTORS)
    selectors: list[SelectorSettings] = []
    for name, value in raw.items():
        key = str(name).strip()
        if key not in known:
            raise AutoTrainConfigError(
                f"selectors.{key!r} is not a known selector. Known selectors: "
                + ", ".join(sorted(known))
            )
        options = _as_mapping(value, f"selectors.{key}")
        enabled = _as_bool(options.get("enabled", False), f"selectors.{key}.enabled")
        if enabled and key in PLANNED_SELECTORS:
            raise AutoTrainConfigError(
                f"selectors.{key} is reserved for a later phase and cannot be "
                "enabled yet."
            )
        selectors.append(
            SelectorSettings(
                name=key,
                enabled=enabled,
                options={
                    str(k): v for k, v in options.items() if str(k) != "enabled"
                },
            )
        )
    selectors.sort(key=lambda item: item.name)
    return tuple(selectors)


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AutoTrainConfigError(f"{label} must be a mapping.")
    return value


def _as_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    raise AutoTrainConfigError(f"{label} must be true or false, got {value!r}.")


def _as_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AutoTrainConfigError(f"{label} must be an integer, got {value!r}.")
    if minimum is not None and value < minimum:
        raise AutoTrainConfigError(f"{label} must be at least {minimum}, got {value}.")
    return value


def _as_float(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AutoTrainConfigError(f"{label} must be a number, got {value!r}.")
    result = float(value)
    if minimum is not None and result < minimum:
        raise AutoTrainConfigError(f"{label} must be at least {minimum}, got {result}.")
    return result


def _as_str_tuple(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise AutoTrainConfigError(f"{label} must be a list of strings.")
    return tuple(str(item).strip() for item in value if str(item).strip())
