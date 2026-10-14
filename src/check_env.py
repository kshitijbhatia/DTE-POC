"""Environment sanity check: imports every pinned library and prints its version.

Run with:  .venv/bin/python -m src.check_env
"""

import importlib
import sys

PACKAGES: list[str] = [
    "numpy",
    "pandas",
    "sklearn",
    "lightgbm",
    "tensorflow",
    "keras",
    "matplotlib",
    "fastapi",
    "uvicorn",
    "openpyxl",
]


def main() -> int:
    """Import each package, print its version, and return a non-zero exit code on failure."""
    print(f"python      {sys.version.split()[0]}")
    failed = []
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"{name:<11} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            print(f"{name:<11} FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}")
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
