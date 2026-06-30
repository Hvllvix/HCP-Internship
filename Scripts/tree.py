"""Build repository directory tree and persist to Others/tree.txt."""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "Others" / "tree.txt"

SKIP = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".cursor",
    "References",
}

SIMPUTE_TREE = """
└── Simpute/  (companion project — PyPI: simpute)
    ├── .github/workflows/release.yml
    ├── simpute/
    │   ├── __init__.py
    │   ├── core.py
    │   ├── models.py
    │   └── utils.py
    ├── tests/
    │   ├── __init__.py
    │   ├── data/test.csv
    │   ├── data/README.md
    │   ├── guard.py
    │   └── mask.py
    ├── Assets/Plots/
    ├── scripts/generate_plots.py
    ├── pyproject.toml
    ├── setup.py
    ├── README.md
    └── LICENSE
"""


def build_tree(dirpath, prefix=""):
    try:
        items = sorted(os.listdir(dirpath))
    except PermissionError:
        return []

    items = [i for i in items if i not in SKIP and not i.endswith(".pyc")]
    lines = []
    for idx, item in enumerate(items):
        path = os.path.join(dirpath, item)
        last = idx == len(items) - 1
        conn = "└── " if last else "├── "
        lines.append(f"{prefix}{conn}{item}")
        if os.path.isdir(path):
            ext = "    " if last else "│   "
            lines.extend(build_tree(path, prefix + ext))
    return lines


def write_tree():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = ["."] + build_tree(str(ROOT)) + [SIMPUTE_TREE.rstrip()]
    OUT.write_text("\n".join(body) + "\n", encoding="utf-8")
    return OUT


if __name__ == "__main__":
    path = write_tree()
    print(f"Tree written to {path}")
