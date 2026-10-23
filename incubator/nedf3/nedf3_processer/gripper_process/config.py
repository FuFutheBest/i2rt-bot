import json
from dataclasses import dataclass


@dataclass
class Cfg:
    hsv_lower: tuple = (43, 50, 50)
    hsv_upper: tuple = (83, 255, 255)
    roi_padding: int = 25
    target_mask_index: int = 2
    min_contour_area: float = 0.5
    min_circularity: float = 0.0
    min_compactness: float = 0.4
    min_brightness_gradient: float = 5.0
    gate_radius: float = 30.0
    max_dropout_frames: int = 30
    erode_px: int = 1
    kalman_sigma: float = 30.0

    @classmethod
    def from_json_or_default(cls, json_str: str) -> "Cfg":
        if not json_str:
            return cls()
        try:
            d = json.loads(json_str)
            if "hsv_lower" in d:
                d["hsv_lower"] = tuple(d["hsv_lower"])
            if "hsv_upper" in d:
                d["hsv_upper"] = tuple(d["hsv_upper"])
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError):
            return cls()

