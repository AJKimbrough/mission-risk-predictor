import os
from typing import Tuple, List
from joblib import load

_bundle = None


def load_model(path: str = os.getenv("SKYSAFE_MODEL", "models/logreg.joblib")):
    global _bundle
    if _bundle is None:
        _bundle = load(path)
    return _bundle["model"], _bundle["features"]