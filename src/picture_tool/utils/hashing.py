import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

def compute_dir_hash(directory: Path, glob_pattern: str = "**/*") -> str:
    """Compute a single hash for a directory structure (filenames + sizes + mtimes)."""
    if not directory.exists():
        return "empty"
    
    hasher = hashlib.md5()
    # Sort for determinism
    files = sorted([p for p in directory.glob(glob_pattern) if p.is_file()])
    
    for p in files:
        try:
            stat = p.stat()
            # Feed path, size, and mtime
            hasher.update(str(p.relative_to(directory)).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime).encode("utf-8"))
        except Exception:
            continue
            
    return hasher.hexdigest()

def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute hash of a configuration dictionary."""
    try:
        # sort_keys=True ensures consistent JSON serialization
        s = json.dumps(config, sort_keys=True)
        return hashlib.md5(s.encode("utf-8")).hexdigest()
    except Exception:
        return "unknown"
