import sys
from pathlib import Path


def ensure_repo_root_on_path():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    for path in (src_dir, repo_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
