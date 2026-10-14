"""Send every example in server/examples/ to a running API and compare with the expected answers.

Start the server first (in another terminal):
    .venv/bin/uvicorn server.app:app --port 8000
Then run:
    .venv/bin/python -m server.try_api
"""

import json
import sys
from pathlib import Path

import httpx

EX_DIR = Path(__file__).resolve().parent / "examples"
URL = "http://127.0.0.1:8000"


def main() -> int:
    """POST each example and print predicted vs expected vs actual."""
    expected = json.loads((EX_DIR / "expected.json").read_text())
    try:
        print("health:", httpx.get(f"{URL}/health", timeout=5).json())
    except httpx.TransportError:
        print(f"Cannot reach {URL}. Start the server first: .venv/bin/uvicorn server.app:app --port 8000")
        return 1
    print(f"\n{'example':<18} {'rate (API)':>10} {'expected':>9} {'actual':>7} | {'DTE API':>8} {'baseline':>9} {'actual':>7}")
    for name, exp in expected.items():
        body = json.loads((EX_DIR / f"{name}.json").read_text())
        r = httpx.post(f"{URL}/predict", json=body, timeout=10)
        r.raise_for_status()
        out = r.json()
        print(f"{name:<18} {out['rate_pct_soc_per_km']:>10.3f} {exp['expected_server_rate_pct_soc_per_km']:>9.3f} "
              f"{exp['actual_rate_next_5km']:>7.3f} | {out['dte_km']:>6.1f}km {out['baseline_dte_km']:>7.1f}km "
              f"{exp['actual_dte_km_if_next_5km_continued']:>5.1f}km")
    return 0


if __name__ == "__main__":
    sys.exit(main())
