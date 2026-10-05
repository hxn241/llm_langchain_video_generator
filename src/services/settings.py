"""config.yaml: the pipeline's settings (everything except keys, which live in the .env)."""
from functools import cache
from pathlib import Path

import yaml

from llmkit import config

CONFIG_FILE = config.BASE / "config.yaml"


@cache
def load_settings(path=CONFIG_FILE):
    """config.yaml as a dict (read once per program). Do not modify the result."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {config.readable_path(path)} (see the README, section Configuración)")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def runs_dir(run_id):
    """<STATE_DIR>/runs/<run_id>/: everything a run produces (script, audio, visuals, final.mp4)."""
    folder = config.STATE_DIR / "runs" / str(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder
