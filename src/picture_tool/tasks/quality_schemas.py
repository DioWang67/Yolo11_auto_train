from typing import Tuple, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field

class SamConfig(BaseModel):
    checkpoint: str = Field(..., description="Path to the SAM checkpoint.")
    model_type: str = Field("vit_b")
    device: str = Field("auto")
    max_side: int = Field(2048)

class ColorInspectionConfig(BaseModel):
    enabled: bool = Field(True)
    input_dir: Path = Field(Path("./data/project/qc/color_samples"))
    output_json: Path = Field(Path("./runs/project/quality/color/stats.json"))
    colors: List[str] = Field(default_factory=list)
    sam: SamConfig
    max_side: int = Field(2048)

class StripSamplingConfig(BaseModel):
    enabled: bool = Field(False)
    segments: int = Field(10)
    threshold: float = Field(0.25)
    orientation: str = Field("vertical")
    min_width_ratio: float = Field(0.05)
    edge_margin: float = Field(0.02)
    sat_threshold: float = Field(40.0)
    val_threshold: float = Field(40.0)
    center_bias: bool = Field(True)
    center_sigma: float = Field(0.3)
    min_valid_pixels: int = Field(100)
    top_k: int = Field(3)
    min_sat_ratio: float = Field(0.05)
    max_edge_ratio: float = Field(0.3)
    black_s_threshold: float = Field(30.0)
    black_v_threshold: float = Field(30.0)

class ColorVerificationConfig(BaseModel):
    enabled: bool = Field(True)
    input_dir: Path = Field(Path("./data/project/qc/color_samples"))
    color_stats: Path = Field(Path("./runs/project/quality/color/stats.json"))
    output_json: Path = Field(Path("./runs/project/quality/color/verification.json"))
    output_csv: Path = Field(Path("./runs/project/quality/color/verification.csv"))
    recursive: bool = Field(False)
    expected_map: Optional[Path] = Field(None)
    expected_from_name: bool = Field(True)
    min_area_ratio: float = Field(0.01)
    max_area_ratio: float = Field(0.8)
    hsv_margin: Tuple[float, float, float] = Field((8.0, 35.0, 40.0))
    lab_margin: Tuple[float, float, float] = Field((12.0, 8.0, 12.0))
    debug_plot: bool = Field(False)
    debug_dir: Optional[Path] = Field(None)
    mask_strategy: str = Field("auto")
    strip_sampling: StripSamplingConfig = Field(default_factory=StripSamplingConfig)
    
    # Optional overrides from root of config
    segments: Optional[int] = Field(None)
    orientation: Optional[str] = Field(None)
    min_strip_ratio: Optional[float] = Field(None)
    ratio_threshold: Optional[float] = Field(None)
    edge_margin: Optional[float] = Field(None)
    sat_threshold: Optional[float] = Field(None)
    val_threshold: Optional[float] = Field(None)
    min_sat_ratio: Optional[float] = Field(None)
    max_edge_ratio: Optional[float] = Field(None)
    black_s_threshold: Optional[float] = Field(None)
    black_v_threshold: Optional[float] = Field(None)
