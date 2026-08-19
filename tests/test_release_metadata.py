from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_latest_released_changelog() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_match = re.search(r'^version = "(\d+\.\d+\.\d+)"$', pyproject, re.MULTILINE)
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    released_versions = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", changelog, re.MULTILINE)

    assert package_match
    assert released_versions
    assert package_match.group(1) == released_versions[0]
