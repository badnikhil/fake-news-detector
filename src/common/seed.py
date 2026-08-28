"""Fix every RNG we use so `make data` / `make train` are reproducible (seed 42 by default)."""
from __future__ import annotations

import os
import random

from src.config import SEED


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    # torch / transformers are seeded only if the caller already imported them: importing torch costs
    # ~0.5 GB RSS and several seconds, which `make data` (pure pandas/sklearn) must not pay.
    import sys

    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if "transformers" in sys.modules:
        sys.modules["transformers"].set_seed(seed)
