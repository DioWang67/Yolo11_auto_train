#!/usr/bin/env python3
"""Safely rewrite absolute project roots after an intentional workspace move.

The command is a dry run unless ``--apply`` is supplied.  It never moves files;
it only rewrites path prefixes in allowlisted text files under the two *new*
project roots.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


TEXT_EXTENSIONS = frozenset(
    {
        ".bat",
        ".cfg",
        ".cmd",
        ".conf",
        ".css",
        ".csv",
        ".html",
        ".ini",
        ".js",
        ".json",
        ".jsonl",
        ".jsx",
        ".md",
        ".ps1",
        ".py",
        ".pyi",
        ".qss",
        ".rst",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsv",
        ".tsx",
        ".txt",
        ".ui",
        ".vue",
        ".xml",
        ".yaml",
        ".yml",
    }
)

SKIPPED_DIRECTORY_NAMES = frozenset(
    {
        ".cache",
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "cache",
        "caches",
        "dist",
        "env",
        "node_modules",
        "venv",
    }
)

# A path root may be followed by a child separator or by syntax that terminates
# a scalar/string.  Rejecting ordinary filename characters prevents changing a
# sibling such as ``C:\\app-backup`` when the old root is ``C:\\app``.
_PREFIX_BOUNDARIES = frozenset(b"/\\\\ \t\r\n\"'`):]},;,#?")


class MigrationError(RuntimeError):
    """Raised when migration inputs or filesystem state are unsafe."""


@dataclass(frozen=True)
class RootMapping:
    """An old project root and its new on-disk location."""

    name: str
    old_root: Path
    new_root: Path


@dataclass(frozen=True)
class Replacement:
    """One byte representation of a root mapping."""

    mapping_name: str
    representation: str
    old: bytes
    new: bytes


@dataclass(frozen=True)
class FileSignature:
    """Fields used to detect edits between planning and atomic replacement."""

    size: int
    modified_ns: int


@dataclass(frozen=True)
class FileChange:
    """A fully planned file rewrite."""

    path: Path
    updated_bytes: bytes
    original_signature: FileSignature
    replacements: tuple[tuple[str, int], ...]

    @property
    def replacement_count(self) -> int:
        return sum(count for _, count in self.replacements)


@dataclass(frozen=True)
class MigrationPlan:
    """Immutable audit plan created before any write occurs."""

    mappings: tuple[RootMapping, ...]
    scanned_files: int
    changes: tuple[FileChange, ...]

    @property
    def replacement_count(self) -> int:
        return sum(change.replacement_count for change in self.changes)


@dataclass(frozen=True)
class MigrationAudit:
    """Result returned by dry-run and apply modes."""

    scanned_files: int
    matched_files: int
    replacement_count: int
    updated_files: int
    changed_paths: tuple[Path, ...]


def _canonical_absolute_path(value: str | Path, *, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        raise MigrationError(f"{label} must be an absolute path: {candidate}")
    try:
        return candidate.resolve(strict=False)
    except OSError as exc:
        raise MigrationError(f"cannot resolve {label} {candidate}: {exc}") from exc


def _is_same_or_nested(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _normalized_identity(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def validate_root_mappings(
    *,
    old_training_root: str | Path,
    new_training_root: str | Path,
    old_inference_root: str | Path,
    new_inference_root: str | Path,
) -> tuple[RootMapping, RootMapping]:
    """Validate and normalize the two requested project-root mappings."""

    training = RootMapping(
        name="training",
        old_root=_canonical_absolute_path(old_training_root, label="old training root"),
        new_root=_canonical_absolute_path(new_training_root, label="new training root"),
    )
    inference = RootMapping(
        name="inference",
        old_root=_canonical_absolute_path(old_inference_root, label="old inference root"),
        new_root=_canonical_absolute_path(new_inference_root, label="new inference root"),
    )
    mappings = (training, inference)

    for mapping in mappings:
        if not mapping.new_root.is_dir():
            raise MigrationError(
                f"new {mapping.name} root does not exist or is not a directory: "
                f"{mapping.new_root}"
            )
        if mapping.new_root.parent == mapping.new_root:
            raise MigrationError(
                f"new {mapping.name} root must not be a filesystem root: {mapping.new_root}"
            )
        if _normalized_identity(mapping.old_root) == _normalized_identity(mapping.new_root):
            raise MigrationError(
                f"old and new {mapping.name} roots must be different: {mapping.old_root}"
            )
        if _is_same_or_nested(mapping.new_root, mapping.old_root):
            raise MigrationError(
                f"new {mapping.name} root must not be inside its old root; this would "
                f"make repeated runs non-idempotent: {mapping.new_root}"
            )

    if _normalized_identity(training.old_root) == _normalized_identity(inference.old_root):
        raise MigrationError("old training and inference roots must be different")
    if _is_same_or_nested(training.old_root, inference.old_root) or _is_same_or_nested(
        inference.old_root, training.old_root
    ):
        raise MigrationError("old training and inference roots must not overlap")

    if _normalized_identity(training.new_root) == _normalized_identity(inference.new_root):
        raise MigrationError("new training and inference roots must be different")
    if _is_same_or_nested(training.new_root, inference.new_root) or _is_same_or_nested(
        inference.new_root, training.new_root
    ):
        raise MigrationError("new training and inference roots must not overlap")

    replacements = build_replacements(mappings)
    old_tokens = tuple(replacement.old for replacement in replacements)
    for replacement in replacements:
        if any(old_token in replacement.new for old_token in old_tokens):
            raise MigrationError(
                "a new root contains an old root representation; repeated runs would "
                "not be idempotent"
            )

    return training, inference


def _path_representations(path: Path) -> tuple[tuple[str, bytes], ...]:
    native = str(path)
    windows = native.replace("/", "\\")
    posix = native.replace("\\", "/")
    escaped_windows = windows.replace("\\", "\\\\")
    return (
        ("escaped-windows", escaped_windows.encode("utf-8")),
        ("windows", windows.encode("utf-8")),
        ("posix", posix.encode("utf-8")),
    )


def build_replacements(mappings: Iterable[RootMapping]) -> tuple[Replacement, ...]:
    """Build deterministic raw/escaped Windows and forward-slash mappings."""

    replacements: list[Replacement] = []
    seen_old_tokens: set[bytes] = set()
    for mapping in mappings:
        old_representations = _path_representations(mapping.old_root)
        new_by_name = dict(_path_representations(mapping.new_root))
        for representation, old_token in old_representations:
            if old_token in seen_old_tokens:
                continue
            seen_old_tokens.add(old_token)
            replacements.append(
                Replacement(
                    mapping_name=mapping.name,
                    representation=representation,
                    old=old_token,
                    new=new_by_name[representation],
                )
            )
    return tuple(replacements)


def _replace_root_prefix(data: bytes, old: bytes, new: bytes) -> tuple[bytes, int]:
    """Replace exact root prefixes without touching similarly named siblings."""

    if not old or old not in data:
        return data, 0

    chunks: list[bytes] = []
    cursor = 0
    count = 0
    while True:
        match_start = data.find(old, cursor)
        if match_start < 0:
            chunks.append(data[cursor:])
            break
        match_end = match_start + len(old)
        is_boundary = match_end == len(data) or data[match_end] in _PREFIX_BOUNDARIES
        if not is_boundary:
            chunks.append(data[cursor:match_end])
            cursor = match_end
            continue
        chunks.append(data[cursor:match_start])
        chunks.append(new)
        cursor = match_end
        count += 1
    return b"".join(chunks), count


def transform_bytes(
    data: bytes, replacements: Sequence[Replacement]
) -> tuple[bytes, tuple[tuple[str, int], ...]]:
    """Apply byte-only replacements and return per-project audit counts."""

    updated = data
    mapping_counts: dict[str, int] = {}
    for replacement in replacements:
        updated, count = _replace_root_prefix(updated, replacement.old, replacement.new)
        if count:
            mapping_counts[replacement.mapping_name] = (
                mapping_counts.get(replacement.mapping_name, 0) + count
            )
    return updated, tuple(sorted(mapping_counts.items()))


def iter_text_files(root: Path) -> Iterable[Path]:
    """Yield allowlisted regular files beneath a root without following links."""

    for current_dir, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current_path = Path(current_dir)
        directory_names[:] = sorted(
            directory_name
            for directory_name in directory_names
            if directory_name.lower() not in SKIPPED_DIRECTORY_NAMES
            and not (current_path / directory_name).is_symlink()
        )
        for file_name in sorted(file_names):
            path = current_path / file_name
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            if path.is_symlink() or not path.is_file():
                continue
            yield path


def _file_signature(path: Path) -> FileSignature:
    stat_result = path.stat()
    return FileSignature(size=stat_result.st_size, modified_ns=stat_result.st_mtime_ns)


def build_migration_plan(mappings: Sequence[RootMapping]) -> MigrationPlan:
    """Read and transform every candidate before any apply-mode write occurs."""

    replacements = build_replacements(mappings)
    changes: list[FileChange] = []
    scanned_files = 0

    for mapping in mappings:
        for path in iter_text_files(mapping.new_root):
            scanned_files += 1
            try:
                signature_before = _file_signature(path)
                original_bytes = path.read_bytes()
                signature_after = _file_signature(path)
            except OSError as exc:
                raise MigrationError(f"cannot read candidate file {path}: {exc}") from exc
            if signature_before != signature_after:
                raise MigrationError(f"candidate changed while it was being read: {path}")

            updated_bytes, counts = transform_bytes(original_bytes, replacements)
            if not counts:
                continue
            changes.append(
                FileChange(
                    path=path,
                    updated_bytes=updated_bytes,
                    original_signature=signature_after,
                    replacements=counts,
                )
            )

    return MigrationPlan(
        mappings=tuple(mappings),
        scanned_files=scanned_files,
        changes=tuple(changes),
    )


def _atomic_replace_file(change: FileChange) -> None:
    path = change.path
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.workspace-migration-",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(change.updated_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        shutil.copystat(path, temporary_path, follow_symlinks=False)
        if _file_signature(path) != change.original_signature:
            raise MigrationError(f"file changed after planning; refusing to overwrite: {path}")
        os.replace(temporary_path, path)
        temporary_path = None
    except MigrationError:
        raise
    except OSError as exc:
        raise MigrationError(f"cannot atomically replace {path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _display_path(path: Path, mappings: Sequence[RootMapping]) -> str:
    for mapping in mappings:
        try:
            relative = path.relative_to(mapping.new_root)
        except ValueError:
            continue
        return f"{mapping.name}:{relative.as_posix()}"
    return str(path)


def print_plan(plan: MigrationPlan, *, apply: bool) -> None:
    """Print a deterministic human-readable audit before writes."""

    print(f"Mode: {'APPLY' if apply else 'DRY-RUN'}")
    for mapping in plan.mappings:
        print(f"Root [{mapping.name}]: {mapping.old_root} -> {mapping.new_root}")
    print(f"Allowlisted files scanned: {plan.scanned_files}")
    print(f"Files with path changes: {len(plan.changes)}")
    print(f"Path-prefix replacements: {plan.replacement_count}")
    for change in plan.changes:
        counts = ", ".join(f"{name}={count}" for name, count in change.replacements)
        action = "UPDATE" if apply else "WOULD UPDATE"
        print(
            f"{action}: {_display_path(change.path, plan.mappings)} "
            f"({counts}; total={change.replacement_count})"
        )


def migrate_workspace_paths(
    *,
    old_training_root: str | Path,
    new_training_root: str | Path,
    old_inference_root: str | Path,
    new_inference_root: str | Path,
    apply: bool = False,
    emit_audit: bool = True,
) -> MigrationAudit:
    """Plan and optionally apply a workspace path migration."""

    mappings = validate_root_mappings(
        old_training_root=old_training_root,
        new_training_root=new_training_root,
        old_inference_root=old_inference_root,
        new_inference_root=new_inference_root,
    )
    plan = build_migration_plan(mappings)
    if emit_audit:
        print_plan(plan, apply=apply)

    updated_files = 0
    if apply:
        for change in plan.changes:
            _atomic_replace_file(change)
            updated_files += 1
        if emit_audit:
            print(f"Files atomically updated: {updated_files}")
    elif emit_audit:
        print("No files were written. Re-run with --apply to commit this exact operation.")

    return MigrationAudit(
        scanned_files=plan.scanned_files,
        matched_files=len(plan.changes),
        replacement_count=plan.replacement_count,
        updated_files=updated_files,
        changed_paths=tuple(change.path for change in plan.changes),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite old training/inference absolute roots under already-moved project "
            "trees. Dry-run is the default; this command never moves directories."
        )
    )
    parser.add_argument("--old-training-root", required=True)
    parser.add_argument("--new-training-root", required=True)
    parser.add_argument("--old-inference-root", required=True)
    parser.add_argument("--new-inference-root", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="atomically rewrite matched files (without this flag, only print a dry run)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        migrate_workspace_paths(
            old_training_root=args.old_training_root,
            new_training_root=args.new_training_root,
            old_inference_root=args.old_inference_root,
            new_inference_root=args.new_inference_root,
            apply=args.apply,
        )
    except MigrationError as exc:
        print(f"Migration refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
