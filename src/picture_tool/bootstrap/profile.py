"""What a station expects, as data rather than as code.

Everything specific to Cable1/A --- two Black wires, one each of the others,
six objects in frame, Red and Orange being the pair that actually gets
confused --- lives in a profile loaded from the station's own configuration.
None of it appears in the bootstrapper's logic.

That separation is the whole reason this module exists. A bootstrapper with
``if class_name == "Red"`` in it works for exactly one station and quietly
mislabels the next one, and the failure looks like a model problem rather
than a hard-coded assumption. So the core asks the profile questions ---
"how many of this class do you expect", "which classes do you confuse" ---
and the profile answers from configuration.

The counts come from ``expected_items``, which is the station's multiset of
expected objects and lists Black twice because the fixture physically has
two black wires. It is emphatically not a class schema and is never used as
one here; see :mod:`picture_tool.autotrain.class_schema` for why that
distinction has already caused trouble.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from picture_tool.autotrain import AutoTrainError
from picture_tool.autotrain.class_schema import (
    ClassSchema,
    resolve_class_schema,
    schema_from_station_config,
)


class ProfileError(AutoTrainError):
    """Raised when a station profile cannot be resolved."""


@dataclass(frozen=True)
class ProductProfile:
    """One station's expectations, resolved from its configuration."""

    product: str
    area: str
    class_schema: ClassSchema
    #: How many instances of each class a good image contains.
    expected_counts: Mapping[str, int] = field(default_factory=dict)
    #: Class pairs this station actually confuses, as unordered pairs. A
    #: disagreement inside one of these is treated as a real ambiguity
    #: rather than noise, because at this station it usually is: Red->Orange
    #: is the single largest human correction in the production record.
    confusion_pairs: tuple[tuple[str, str], ...] = ()
    #: Fraction of the box to sample when measuring colour, so the reading
    #: comes from the object rather than from the background around it.
    colour_roi_inset: tuple[float, float] = (0.2, 0.0)
    #: Reference colour statistics, distilled from images a person confirmed.
    colour_stats_path: Path | None = None

    @property
    def expected_total(self) -> int:
        return sum(self.expected_counts.values())

    def is_confusable(self, left: str, right: str) -> bool:
        pair = frozenset({left, right})
        return any(frozenset(candidate) == pair for candidate in self.confusion_pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "area": self.area,
            "class_schema": self.class_schema.to_dict(),
            "expected_counts": dict(self.expected_counts),
            "expected_total": self.expected_total,
            "confusion_pairs": [list(pair) for pair in self.confusion_pairs],
            "colour_roi_inset": list(self.colour_roi_inset),
            "colour_stats_path": (
                str(self.colour_stats_path) if self.colour_stats_path else None
            ),
        }


def expected_counts_from_items(items: Sequence[Any]) -> dict[str, int]:
    """Count a station's expected-object multiset.

    Counting ``expected_items`` is the one legitimate use for it. Reading it
    as an ordered class list is not: its order differs from the contract and
    it repeats a name, which is exactly how a class id ends up meaning the
    wrong colour.
    """
    counts: collections.Counter[str] = collections.Counter()
    for item in items:
        name = str(item).strip()
        if name:
            counts[name] += 1
    return dict(counts)


def load_profile(
    model_dir: str | Path,
    *,
    product: str,
    area: str,
    champion_schema: ClassSchema | None = None,
    confusion_pairs: Sequence[Sequence[str]] | None = None,
) -> ProductProfile:
    """Resolve a station profile from the deployed station configuration.

    Read-only over the production model directory: this opens the config a
    station already runs on and copies nothing back.

    ``confusion_pairs`` is a caller-supplied hint rather than something
    inferred here. Deriving it from production corrections would be
    reasonable and is deliberately not done in this pass --- a profile that
    changes shape as new data arrives is a moving definition, and this POC
    needs a fixed one to measure against.
    """
    directory = Path(model_dir)
    config_path = directory / "config.yaml"
    if not config_path.is_file():
        raise ProfileError(f"No station configuration at {config_path}.")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ProfileError(f"Could not read {config_path}: {exc}") from exc

    schema = resolve_class_schema(
        [champion_schema, schema_from_station_config(directory)],
        context=f"{product}/{area}",
    )

    items = (payload.get("expected_items") or {}).get(product, {}).get(area)
    counts = expected_counts_from_items(items) if isinstance(items, list) else {}
    unknown = sorted(set(counts) - set(schema.names))
    if unknown:
        raise ProfileError(
            f"{config_path} expects items {unknown} that the class contract "
            f"{list(schema.names)} does not contain. One of the two is wrong, "
            "and guessing which would put boxes on the wrong class."
        )

    policy = payload.get("color_roi_policy") or {}
    inset = (
        float(policy.get("inset_x_ratio", 0.2) or 0.0),
        float(policy.get("inset_y_ratio", 0.0) or 0.0),
    )

    stats_relative = str(payload.get("color_model_path") or "").strip()
    stats_path: Path | None = None
    if stats_relative:
        # Station configs record this relative to the inference project root,
        # which is the models/ directory's grandparent.
        for candidate in (
            directory / Path(stats_relative).name,
            _inference_root(directory) / stats_relative,
        ):
            if candidate.is_file():
                stats_path = candidate
                break

    return ProductProfile(
        product=product,
        area=area,
        class_schema=schema,
        expected_counts=counts,
        confusion_pairs=tuple(
            (str(pair[0]), str(pair[1]))
            for pair in (confusion_pairs or ())
            if len(pair) == 2
        ),
        colour_roi_inset=inset,
        colour_stats_path=stats_path,
    )


def _inference_root(model_dir: Path) -> Path:
    for parent in model_dir.resolve().parents:
        if parent.name == "models":
            return parent.parent
    return model_dir.resolve()
