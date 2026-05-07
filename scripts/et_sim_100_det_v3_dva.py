import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent

os.environ.setdefault("ET_EFFECT_PROFILE", "v3_dva")
os.environ.setdefault("ET_PROFILE_TARGET_FRAMES", "20")
os.environ.setdefault("ET_RUN_ALL_BATCHES", "0")
os.environ.setdefault("ET_OUTPUT_RUN_NAME_OVERRIDE", "v3_v3_dva_20f")

runpy.run_path(str(ROOT / "et_sim_100_det.py"), run_name="__main__")
