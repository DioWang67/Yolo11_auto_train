"""The one place this tool's colours, type and shape are decided.

Two applications sit in front of the same operator: the inference GUI on the
line (``yolo11_inference``) and this training tool. They were drifting into
opposite visual languages -- the inference GUI is light, this tool was a dark
GitHub theme -- so an operator moving between them had to re-learn which colour
meant "go" and which meant "stop". The tokens below are the inference GUI's own
values, read out of its stylesheets, so the two now agree.

Why a module instead of a ``.qss`` file: the colours are needed from Python as
well. Roughly forty ``setStyleSheet`` calls across this package set a label's
colour or a card's background inline, and while those literals lived beside
each stylesheet the two could disagree -- which is exactly how the dark values
survived. Tokens here are the single source for both, and
``build_stylesheet()`` renders the application-wide sheet from them.

Scope is deliberately narrow. The inference GUI styles only ``QPushButton``,
``QComboBox``, ``QGroupBox`` and the window/font defaults, leaving line edits,
check boxes, tabs, tables and scroll bars to Qt's native light rendering. This
sheet covers the same ground and no more, because a rule with no counterpart
over there is a new way for the two to look different. What it adds are the
object names this tool has and the inference GUI does not -- its sidebar, the
config editor surfaces, the operator workflow panel -- rebuilt from these same
tokens.

Source of each value, for anyone checking a drift:
``yolo11_inference/app/gui/main_window.py`` (window, font, buttons, combo box,
group box), ``app/gui/panels/info_panel.py`` (log console),
``app/gui/widgets.py`` (status washes, content text area),
``app/gui/color_preflight_dialog.py`` (OK/WARN/NG),
``app/gui/review_selection_gallery.py`` (list).
"""

from __future__ import annotations

from typing import NamedTuple

# --------------------------------------------------------------------------
# Surfaces
# --------------------------------------------------------------------------
#: Application background, behind every page.
SURFACE_APP = "#f4f6f8"
#: Cards, group boxes, dialogs -- anything holding content.
SURFACE_CARD = "#ffffff"
#: Default button face and other quiet raised surfaces.
SURFACE_MUTED = "#eef2f6"
#: Log and other monospace read-only views.
SURFACE_CONSOLE = "#f8f9fa"
#: List backgrounds, one step off the card.
SURFACE_LIST = "#f6f8fb"
#: Disabled control face.
SURFACE_DISABLED = "#d9e2ec"

# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------
TEXT_PRIMARY = "#1f2933"
TEXT_HEADING = "#334e68"
#: Hints, secondary lines, path previews -- present but not competing.
TEXT_MUTED = "#6b7280"
#: On a filled primary/danger action.
TEXT_ON_ACCENT = "#ffffff"
TEXT_DISABLED = "#829ab1"
#: Body text inside a list item.
TEXT_LIST_ITEM = "#243447"

# --------------------------------------------------------------------------
# Banner
#
# The inference GUI puts prominent instructions on a dark band with white
# text -- the same declaration appears verbatim at the top of four of its
# dialogs. It is the one deliberately dark surface in an otherwise light
# application, so a light "notice" treatment here would read as a different
# kind of message than the same instruction does on the line.
# --------------------------------------------------------------------------
SURFACE_BANNER = "#243447"
TEXT_ON_BANNER = "#ffffff"

# --------------------------------------------------------------------------
# Borders
# --------------------------------------------------------------------------
BORDER_BUTTON = "#cbd5df"
BORDER_INPUT = "#bcccdc"
#: Group boxes, dividers, and the disabled button's own edge.
BORDER_DIVIDER = "#d9e2ec"
BORDER_CONSOLE = "#dee2e6"
BORDER_LIST = "#d0d5dd"
BORDER_LIST_ITEM = "#c8d0da"
#: Hover emphasis on an otherwise quiet edge.
BORDER_STRONG = "#9fb0c3"

# --------------------------------------------------------------------------
# Actions
#
# The inference GUI has exactly three: a filled green primary, a filled red
# danger, and a white bordered secondary. It has no fourth "success" tier --
# its primary *is* green -- so a second green button would collide with the
# main action rather than rank below it.
# --------------------------------------------------------------------------
ACTION_PRIMARY = "#16794c"
ACTION_PRIMARY_HOVER = "#12643f"
ACTION_DANGER = "#b42318"
ACTION_DANGER_HOVER = "#971d14"
ACTION_SECONDARY_TEXT = "#243b53"
BUTTON_HOVER = "#e4ebf2"
BUTTON_PRESSED = "#d7e1ec"
#: Selection emphasis inside lists, from the inference GUI's gallery.
SELECTION_HOVER = "#5b8fc9"
SELECTION_ACTIVE = "#2563a6"


class SemanticColor(NamedTuple):
    """A status's text, background wash and border, kept together.

    The three are only meaningful as a set -- a wash paired with another
    status's text is how an unreadable combination gets shipped -- so callers
    take the triplet rather than three loose constants.
    """

    text: str
    wash: str
    border: str


#: A check that passed, a valid path, a completed step.
STATUS_OK = SemanticColor("#237a3b", "#edf7ed", "#7bc47f")
#: Something needing attention that does not block.
STATUS_WARN = SemanticColor("#8a5a10", "#fbf1de", "#f0b429")
#: A failure, a blocking issue, a destructive action.
STATUS_NG = SemanticColor("#b42318", "#fbeae8", "#d9534f")
#: Neutral progress and informational notices.
STATUS_INFO = SemanticColor("#245b8f", "#eef6ff", "#c7ddf2")
#: Acknowledgement that carries no status -- "selection cleared", an
#: inactive step. Derived from the tokens above rather than a colour of its
#: own, so a caller with nothing to report still has something to pass.
STATUS_NEUTRAL = SemanticColor(TEXT_MUTED, SURFACE_MUTED, BORDER_DIVIDER)

# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------
RADIUS_CONTROL = "6px"
RADIUS_CARD = "8px"
#: Read-only console and other inset views.
RADIUS_INSET = "4px"

# --------------------------------------------------------------------------
# Type
#
# JhengHei leads, matching the inference GUI. The order is not cosmetic: with
# Segoe UI first -- as this tool had it -- Latin text and digits render from a
# different face than they do on the line, so the same batch number looks like
# two different numbers in two windows.
# --------------------------------------------------------------------------
#: For the Qt APIs that take one family rather than a CSS list --
#: ``QApplication.setFont`` and ``QFont``. The stack below is built from it
#: rather than repeating it: a second literal is exactly how the application
#: font and the stylesheet came to ask for different faces.
FONT_FAMILY_PRIMARY = "Microsoft JhengHei"
FONT_FAMILY = f'"{FONT_FAMILY_PRIMARY}", "Segoe UI", Arial'
FONT_MONO = "Consolas"
FONT_SIZE_HEADING = "12pt"
#: One step above body, for a summary line that should carry weight.
FONT_SIZE_LARGE = "11pt"
FONT_SIZE_BODY = "10pt"
FONT_SIZE_SMALL = "9pt"
FONT_SIZE_TINY = "8pt"


def muted_text(size: str = FONT_SIZE_SMALL) -> str:
    """Return the inline style for a secondary line of text.

    The commonest inline style in this package by a wide margin; naming it
    keeps a dozen call sites from each spelling out the same pair.
    """
    return f"color: {TEXT_MUTED}; font-size: {size};"


def mono_text(color: str = TEXT_MUTED, size: str = FONT_SIZE_TINY) -> str:
    """Return the inline style for a monospace preview line (paths, metrics)."""
    return f"color: {color}; font-size: {size}; font-family: {FONT_MONO};"


def status_text(status: SemanticColor, size: str = FONT_SIZE_SMALL) -> str:
    """Return the inline style for a status line carrying ``status``."""
    return f"color: {status.text}; font-size: {size};"


def log_severity_status(message: str) -> SemanticColor:
    """Return the colours a log line should be rendered in.

    Both log views -- :mod:`picture_tool.gui.log_viewer` and the main
    window's own -- had grown an independent copy of this mapping, which is
    how one of them could end up disagreeing about what a warning looks like.
    It lives here rather than beside either view because it answers a palette
    question, and because this module has no imports of its own to fail: the
    main window resolves its view components through a fallback import, and a
    severity that cannot be coloured would leave log output unreadable
    exactly when something has already gone wrong.
    """
    lower = message.lower()
    if "error" in lower:
        return STATUS_NG
    if "warning" in lower:
        return STATUS_WARN
    if "success" in lower:
        return STATUS_OK
    if "info" in lower:
        return STATUS_INFO
    return STATUS_NEUTRAL


def status_card(status: SemanticColor, radius: str = RADIUS_CARD) -> str:
    """Return the inline style for a card whose surface carries a status."""
    return (
        f"background: {status.wash}; color: {status.text}; "
        f"border: 1px solid {status.border}; border-radius: {radius};"
    )


def build_stylesheet() -> str:
    """Return the application-wide stylesheet.

    Rendered from the tokens above rather than read from disk, so a colour
    cannot be changed in one of the two places that use it.
    """
    return f"""
/* Window and inherited defaults ---------------------------------------- */
QMainWindow, QDialog {{
    background-color: {SURFACE_APP};
}}
QWidget {{
    font-family: {FONT_FAMILY};
    font-size: {FONT_SIZE_BODY};
    color: {TEXT_PRIMARY};
}}

/* Buttons -- the inference GUI's three tiers -------------------------- */
QPushButton {{
    background-color: {SURFACE_MUTED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_BUTTON};
    padding: 8px 12px;
    border-radius: {RADIUS_CONTROL};
    font-weight: 600;
    min-width: 60px;
}}
QPushButton:hover {{
    background-color: {BUTTON_HOVER};
    border-color: {BORDER_STRONG};
}}
QPushButton:pressed {{
    background-color: {BUTTON_PRESSED};
}}
QPushButton:disabled {{
    background-color: {SURFACE_DISABLED};
    color: {TEXT_DISABLED};
    border-color: {SURFACE_DISABLED};
}}

QPushButton#primaryAction {{
    background-color: {ACTION_PRIMARY};
    color: {TEXT_ON_ACCENT};
    border: none;
    padding: 10px 20px;
    min-width: 100px;
}}
QPushButton#primaryAction:hover {{
    background-color: {ACTION_PRIMARY_HOVER};
}}
QPushButton#primaryAction:disabled {{
    background-color: {SURFACE_DISABLED};
    color: {TEXT_DISABLED};
}}

QPushButton#dangerAction {{
    background-color: {ACTION_DANGER};
    color: {TEXT_ON_ACCENT};
    border: none;
}}
QPushButton#dangerAction:hover {{
    background-color: {ACTION_DANGER_HOVER};
}}
QPushButton#dangerAction:disabled {{
    background-color: {SURFACE_DISABLED};
    color: {TEXT_DISABLED};
}}

QPushButton#secondaryAction {{
    background-color: {SURFACE_CARD};
    color: {ACTION_SECONDARY_TEXT};
    border: 1px solid {BORDER_INPUT};
}}
QPushButton#secondaryAction:hover {{
    background-color: {BUTTON_HOVER};
    border-color: {BORDER_STRONG};
}}

/* Combo box ------------------------------------------------------------ */
QComboBox {{
    padding: 6px 10px;
    border: 1px solid {BORDER_INPUT};
    border-radius: {RADIUS_CONTROL};
    background-color: {SURFACE_CARD};
    min-width: 120px;
}}
QComboBox:hover {{
    border-color: {BORDER_STRONG};
}}

/* Group box ------------------------------------------------------------ */
QGroupBox {{
    font-weight: 700;
    border: 1px solid {BORDER_DIVIDER};
    border-radius: {RADIUS_CARD};
    margin-top: 1ex;
    padding-top: 12px;
    background-color: {SURFACE_CARD};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px 0 6px;
    color: {TEXT_HEADING};
}}

/* Read-only text views ------------------------------------------------- */
QTextEdit {{
    background-color: {SURFACE_CONSOLE};
    border: 1px solid {BORDER_CONSOLE};
    border-radius: {RADIUS_INSET};
    padding: 8px;
}}

/* Lists ---------------------------------------------------------------- */
QListWidget {{
    background-color: {SURFACE_LIST};
    border: 1px solid {BORDER_LIST};
    border-radius: {RADIUS_CARD};
    padding: 4px;
}}
QListWidget::item {{
    border: 1px solid {BORDER_LIST_ITEM};
    border-radius: {RADIUS_CONTROL};
    padding: 6px;
    margin: 2px 0px;
    color: {TEXT_LIST_ITEM};
}}
QListWidget::item:hover {{
    border: 2px solid {SELECTION_HOVER};
}}
QListWidget::item:selected {{
    border: 2px solid {SELECTION_ACTIVE};
    color: {TEXT_LIST_ITEM};
}}

/* Progress -- built from the palette; the inference GUI's only progress
   bar is a dark outlier inside its own light app, so it is not the model.

   The chunk is the light green rather than {ACTION_PRIMARY}, because these
   bars show their percentage as text and the fill moves across it. One text
   colour has to stay readable over both the track and the fill: dark text
   measures 13.1:1 on the track and 7.1:1 on this fill, where the darker
   green would give 2.7:1 -- below WCAG AA -- for whatever part of the label
   the fill happens to be under. */
QProgressBar {{
    background-color: {SURFACE_MUTED};
    border: 1px solid {BORDER_DIVIDER};
    border-radius: {RADIUS_INSET};
    min-height: 18px;
    text-align: center;
    color: {TEXT_PRIMARY};
}}
QProgressBar::chunk {{
    background-color: {STATUS_OK.border};
    border-radius: {RADIUS_INSET};
}}

/* This tool's own surfaces -------------------------------------------- */
QWidget#SideBar {{
    background: {SURFACE_CARD};
    border-right: 1px solid {BORDER_DIVIDER};
}}
QScrollArea#SideBarScroll {{
    background: transparent;
    border: none;
}}
QWidget#SideBarScrollContent {{
    background: transparent;
}}

QLabel.Header {{
    font-weight: bold;
    font-size: {FONT_SIZE_HEADING};
    color: {TEXT_HEADING};
    padding: 8px 0px;
    border-bottom: 2px solid {BORDER_DIVIDER};
    margin-bottom: 12px;
}}

QWidget#ConfigEditorTab,
QWidget#ConfigEditorContent {{
    background: {SURFACE_APP};
}}
QScrollArea#ConfigEditorScroll {{
    background: {SURFACE_APP};
    border: none;
}}
QWidget#ConfigEditorContent QLabel {{
    color: {TEXT_MUTED};
    font-size: {FONT_SIZE_BODY};
}}
QWidget#ConfigEditorContent QSpinBox,
QWidget#ConfigEditorContent QDoubleSpinBox {{
    background-color: {SURFACE_CARD};
    border: 1px solid {BORDER_INPUT};
    color: {TEXT_PRIMARY};
    padding: 6px 10px;
    border-radius: {RADIUS_CONTROL};
}}
QWidget#ConfigEditorContent QSpinBox:focus,
QWidget#ConfigEditorContent QDoubleSpinBox:focus {{
    border-color: {BORDER_STRONG};
}}

QDialog#SamDialog QDialogButtonBox QPushButton {{
    min-width: 80px;
}}

/* Horizontal rule (QFrame::HLine) ------------------------------------- */
QFrame[frameShape="4"] {{
    background-color: {BORDER_DIVIDER};
    max-height: 1px;
}}
"""
