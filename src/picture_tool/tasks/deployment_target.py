"""Shared helpers for resolving inference deployment targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from picture_tool.path_resolver import parse_project_area_override


@dataclass(frozen=True)
class DeploymentTarget:
    """Resolved product and area used by inference-ready artifacts.

    Args:
        product: Product directory name under ``models``.
        area: Area directory name under the product.
    """

    product: str
    area: str


def resolve_yolo_deployment_target(
    config: dict[str, Any],
    args: Any,
    *,
    artifact_config_key: str | None = None,
) -> DeploymentTarget:
    """Resolve the YOLO inference target from CLI/GUI override and config.

    Args:
        config: Pipeline configuration.
        args: Runtime args namespace; ``product`` may be ``PRODUCT,AREA``.
        artifact_config_key: Optional ``yolo_training`` subsection to prefer
            before deploy and position validation settings, e.g.
            ``"artifact_bundle"``.

    Returns:
        A deployment target with product and area strings.
    """

    ycfg = config.get("yolo_training", {}) or {}
    preferred_cfg = ycfg.get(artifact_config_key, {}) if artifact_config_key else {}
    deploy_cfg = ycfg.get("deploy", {}) or {}
    position_cfg = ycfg.get("position_validation", {}) or {}

    override = getattr(args, "product", None)
    parsed_override = parse_project_area_override(str(override)) if override else None

    product = (
        parsed_override.project
        if parsed_override
        else preferred_cfg.get("product")
        or deploy_cfg.get("product")
        or position_cfg.get("product")
        or ycfg.get("name", "")
    )
    area = (
        parsed_override.area
        if parsed_override and parsed_override.area
        else preferred_cfg.get("area")
        or deploy_cfg.get("area")
        or position_cfg.get("area")
        or "A"
    )

    return DeploymentTarget(product=str(product), area=str(area))
