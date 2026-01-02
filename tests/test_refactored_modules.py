import pytest
import json
from pathlib import Path
from picture_tool.utils.normalization import normalize_imgsz, normalize_name_sequence
from picture_tool.utils.hashing import compute_dir_hash, compute_config_hash
from picture_tool.constants import DEFAULT_RUNS_DIR

def test_constants():
    assert str(DEFAULT_RUNS_DIR) == "runs"

def test_normalization():
    assert normalize_imgsz(640) == [640, 640]
    assert normalize_name_sequence({"0": "foo"}) == ["foo"]

def test_hashing(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "file.txt").write_text("content", encoding="utf-8")
    hash1 = compute_dir_hash(d)
    
    # modify content
    (d / "file.txt").write_text("content2", encoding="utf-8")
    hash2 = compute_dir_hash(d)
    assert hash1 != hash2

def test_config_hash():
    c1 = {"a": 1, "b": 2}
    c2 = {"b": 2, "a": 1} # Order shouldn't matter if sorted internally
    assert compute_config_hash(c1) == compute_config_hash(c2)
    
    c3 = {"a": 1, "b": 3}
    assert compute_config_hash(c1) != compute_config_hash(c3)
