"""Cross-version support for preserving secondary failure diagnostics."""

from __future__ import annotations


def add_exception_note(error: BaseException, note: str) -> None:
    """Attach a PEP 678-style note on every supported Python version."""
    if not isinstance(note, str):
        raise TypeError(f"note must be a str, not {type(note).__name__}")

    native_add_note = getattr(error, "add_note", None)
    if callable(native_add_note):
        native_add_note(note)
        return

    notes = getattr(error, "__notes__", None)
    if notes is None:
        notes = []
        setattr(error, "__notes__", notes)
    if not isinstance(notes, list):
        raise TypeError("exception __notes__ must be a list")
    notes.append(note)
