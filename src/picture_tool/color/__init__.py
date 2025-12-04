from importlib import import_module

color_inspection = import_module(".color_inspection", __name__)
color_verifier = import_module(".color_verifier", __name__)

__all__ = ["color_inspection", "color_verifier"]
