import sys
from pathlib import Path

# torch/ first, so `import prune` is the torch port, not the MLX prune.py at the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
