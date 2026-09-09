"""Guards on the GUI's single source of colour.

The training tool and the inference GUI on the line are two applications in
front of one operator, and they had drifted into opposite visual languages --
this tool a dark theme, the inference GUI light. Aligning them was a one-off
edit across a dozen files; keeping them aligned is what these tests are for.

The colours themselves are not asserted against the inference GUI. That
repository releases on its own cadence and keeps no single palette to compare
against -- each of its dialogs carries its own status dict -- so a test here
that pinned its hex values would fail on an unrelated change over there and
teach whoever hit it to delete the test. What is enforced instead is the
property that made the drift possible: colours scattered through the widget
code, where nothing can see them all at once.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from picture_tool.gui import theme

#: Every source file that builds part of the tool's interface. ``color`` is
#: included because the SAM selection window lives there and carried a second
#: copy of the application stylesheet, in dark, for exactly as long as nobody
#: was looking for one outside ``gui``.
_UI_PACKAGE_DIRS = ("gui", "color")
#: Vendored third-party UI, not ours to restyle.
_EXCLUDED_PARTS = ("libs", "__pycache__")

_HEX_COLOUR = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")


def _ui_source_files() -> list[Path]:
    package_root = Path(theme.__file__).resolve().parent.parent
    files: list[Path] = []
    for package in _UI_PACKAGE_DIRS:
        for path in sorted((package_root / package).rglob("*.py")):
            if any(part in _EXCLUDED_PARTS for part in path.parts):
                continue
            if path.name == "theme.py":
                continue
            files.append(path)
    return files


def test_ui_sources_carry_no_literal_colours() -> None:
    """No hex colour may appear outside :mod:`picture_tool.gui.theme`.

    This is the check that would have caught the drift while it was one
    label rather than forty, and the one that stops a future "just this one
    label" from starting it again. A new colour belongs in ``theme``, next to
    the others, where its contrast against the surfaces it lands on can be
    judged.
    """
    offenders: list[str] = []
    for path in _ui_source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in _HEX_COLOUR.finditer(line):
                offenders.append(f"{path.name}:{line_number}: {match.group()}")

    assert not offenders, (
        "Literal colours found outside picture_tool.gui.theme. Add the colour "
        "to theme.py and reference it from there:\n  "
        + "\n  ".join(offenders)
    )


def test_ui_sources_do_not_name_the_old_button_variants() -> None:
    """Button object names must use the inference GUI's vocabulary.

    The two applications named the same three tiers differently -- this tool
    said ``PrimaryBtn`` where the line said ``primaryAction`` -- which is how
    a shared design language becomes two. A stale name is worse than
    cosmetic: the stylesheet selects on it, so a renamed button that keeps an
    old name silently loses its styling instead of failing.
    """
    stale = ("PrimaryBtn", "SuccessBtn", "DangerBtn")
    offenders: list[str] = []
    for path in _ui_source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for name in stale:
                if name in line:
                    offenders.append(f"{path.name}:{line_number}: {name}")

    assert not offenders, (
        "Old button object names found; use primaryAction / dangerAction / "
        "secondaryAction:\n  " + "\n  ".join(offenders)
    )


def test_stylesheet_resolves_every_token_it_references() -> None:
    """``build_stylesheet`` must leave no placeholder unrendered.

    The sheet is an f-string over the tokens, so a typo in a name is a
    ``NameError`` at import; what this catches is the subtler outcome of
    doubling a brace wrongly and shipping a literal ``{TOKEN}`` into Qt,
    which Qt ignores in silence.
    """
    sheet = theme.build_stylesheet()

    assert sheet.strip(), "Stylesheet must not be empty."
    leftovers = re.findall(r"\{[A-Za-z_]+\}", sheet)
    assert not leftovers, f"Unrendered stylesheet placeholders: {leftovers}"
    assert "QPushButton#primaryAction" in sheet
    assert "QPushButton#dangerAction" in sheet
    assert "QPushButton#secondaryAction" in sheet


@pytest.mark.parametrize(
    "status",
    [
        theme.STATUS_OK,
        theme.STATUS_WARN,
        theme.STATUS_NG,
        theme.STATUS_INFO,
        theme.STATUS_NEUTRAL,
    ],
)
def test_every_status_is_a_complete_triplet(status: theme.SemanticColor) -> None:
    """A status must supply text, wash and border together.

    Callers style a whole card from one of these. A missing member would not
    raise -- it would render an empty CSS value, which Qt drops, leaving a
    card with the previous widget's background and an unreadable pairing.
    """
    for member in (status.text, status.wash, status.border):
        assert _HEX_COLOUR.fullmatch(member), f"{status} has a non-colour member"


def test_severity_mapping_covers_the_log_levels_it_claims() -> None:
    """Log lines map to the status their wording implies.

    Both log views render through this one function; the mapping used to
    exist twice, once per view.
    """
    assert theme.log_severity_status("[ERROR] boom") is theme.STATUS_NG
    assert theme.log_severity_status("[WARNING] hmm") is theme.STATUS_WARN
    assert theme.log_severity_status("[INFO] fyi") is theme.STATUS_INFO
    assert theme.log_severity_status("Deploy success") is theme.STATUS_OK
    assert theme.log_severity_status("plain line") is theme.STATUS_NEUTRAL


def test_jhenghei_leads_the_font_stack() -> None:
    """The CJK face must come first, as it does on the line.

    With Segoe UI first -- as this tool had it -- Latin glyphs and digits
    render from a different face than they do in the inference GUI, so one
    batch number can look like two different numbers in two windows.
    """
    families = [part.strip().strip('"') for part in theme.FONT_FAMILY.split(",")]
    assert families[0] == "Microsoft JhengHei", families
    # The single-family constant the Qt APIs take must be the same face the
    # stylesheet asks for first, or the application font and the sheet
    # disagree again -- which is the state this replaced.
    assert theme.FONT_FAMILY_PRIMARY == families[0]
