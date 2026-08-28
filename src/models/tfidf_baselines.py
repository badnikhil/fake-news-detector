"""Alias kept because master doc §24 / MSE1 §3.5 name this file; the implementation is ``src.models.baselines``."""
from src.models.baselines import *  # noqa: F401,F403
from src.models.baselines import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
