"""Alias named in the MSE2 doc: the FEVER stance evaluation lives in `src/stance/fever_eval.py`."""
from src.stance.fever_eval import *  # noqa: F401,F403
from src.stance.fever_eval import main

if __name__ == "__main__":
    raise SystemExit(main())
