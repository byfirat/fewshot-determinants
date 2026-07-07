import platform
import sys

import yaml

try:
    import torch
except Exception:
    torch = None

from fewshot_determinants.paths import CONFIGS_DIR

project = yaml.safe_load((CONFIGS_DIR / "project.yaml").read_text(encoding="utf-8"))

print("=== Environment check ===")
print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"Project: {project['project_name']}")
print(f"Seed: {project['seed']}")
if torch is not None:
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
else:
    print("Torch: not installed")
