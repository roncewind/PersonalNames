import json
import os
from dataclasses import dataclass
from typing import Any, Dict


# -----------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    languages: Dict[str, Any]


# -----------------------------------------------------------------------------
def load_config(path: str) -> AugmentConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AugmentConfig(languages=data.get("languages", {}))


# -----------------------------------------------------------------------------
def load_default_config() -> AugmentConfig:
    here = os.path.dirname(__file__)
    path = os.path.join(here, "resources", "default_config.json")
    return load_config(path)
