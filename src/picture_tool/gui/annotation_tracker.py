"""Annotation tracking and validation module."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class AnnotationTracker:
    """Track annotation progress and validate label files."""

    def __init__(self):
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.label_extension = ".txt"

    def scan_directory(
        self,
        image_dir: Path,
        label_dir: Path,
    ) -> Dict:
        """Scan directories and get annotation statistics.
        
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing label files
        
        Returns:
            Dictionary with statistics:
            - total_images: Total number of images
            - annotated_images: Number of annotated images
            - unannotated_images: List of unannotated image names
            - annotated_images_list: List of annotated image names
            - progress_percent: Percentage of annotated images
        """
        if not image_dir.exists():
            logger.warning(f"Image directory does not exist: {image_dir}")
            return self._empty_stats()
        
        # Get all image files
        image_files: set[str] = set()
        for ext in self.image_extensions:
            image_files.update(
                f.stem for f in image_dir.glob(f"*{ext}")
            )
        
        # Get all label files
        label_files = set()
        if label_dir.exists():
            label_files = {
                f.stem for f in label_dir.glob(f"*{self.label_extension}")
            }
        
        # Calculate statistics
        annotated =image_files & label_files
        unannotated = image_files - label_files
        
        total = len(image_files)
        annotated_count = len(annotated)
        progress = (annotated_count / total * 100) if total > 0 else 0
        
        return {
            "total_images": total,
            "annotated_images": annotated_count,
            "unannotated_images": sorted(list(unannotated)),
            "annotated_images_list": sorted(list(annotated)),
            "progress_percent": progress,
        }

    def validate_annotations(
        self,
        label_dir: Path,
        num_classes: int,
    ) -> List[str]:
        """Validate annotation files for errors.
        
        Args:
            label_dir: Directory containing label files
            num_classes: Number of valid classes
        
        Returns:
            List of error messages (empty if all valid)
        """
        errors = []
        
        if not label_dir.exists():
            return [f"Label directory does not exist: {label_dir}"]
        
        label_files = list(label_dir.glob(f"*{self.label_extension}"))
        
        for label_file in label_files:
            file_errors = self._validate_single_file(label_file, num_classes)
            if file_errors:
                errors.extend([f"{label_file.name}: {err}" for err in file_errors])
        
        return errors

    def _validate_single_file(
        self,
        label_file: Path,
        num_classes: int,
    ) -> List[str]:
        """Validate a single label file.
        
        Args:
            label_file: Path to label file
            num_classes: Number of valid classes
        
        Returns:
            List of error messages for this file
        """
        errors = []
        
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    errors.append(
                        f"Line {line_num}: Expected 5 values (class x y w h), got {len(parts)}"
                    )
                    continue
                
                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    
                    # Validate class ID
                    if class_id < 0 or class_id >= num_classes:
                        errors.append(
                            f"Line {line_num}: Invalid class_id {class_id} "
                            f"(must be 0-{num_classes-1})"
                        )
                    
                    # Validate coordinates (should be normalized 0-1)
                    for coord_name, coord_val in [("x", x), ("y", y), ("w", w), ("h", h)]:
                        if not (0 <= coord_val <= 1):
                            errors.append(
                                f"Line {line_num}: {coord_name}={coord_val} "
                                f"out of range [0, 1]"
                            )
                
                except ValueError as e:
                    errors.append(f"Line {line_num}: Cannot parse values - {e}")
        
        except Exception as e:
            errors.append(f"Cannot read file: {e}")
        
        return errors

    def get_class_distribution(
        self,
        label_dir: Path,
        class_names: List[str],
    ) -> Dict[str, int]:
        """Get distribution of classes across all annotations.
        
        Args:
            label_dir: Directory containing label files
            class_names: List of class names (indexed by class_id)
        
        Returns:
            Dictionary mapping class names to counts
        """
        if not label_dir.exists():
            return {}
        
        class_counts: Counter = Counter()
        
        for label_file in label_dir.glob(f"*{self.label_extension}"):
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(class_names):
                                class_counts[class_names[class_id]] += 1
            except Exception as e:
                logger.warning(f"Error reading {label_file}: {e}")
                continue
        
        return dict(class_counts)

    def _empty_stats(self) -> Dict:
        """Return empty statistics dictionary."""
        return {
            "total_images": 0,
            "annotated_images": 0,
            "unannotated_images": [],
            "annotated_images_list": [],
            "progress_percent": 0.0,
        }
