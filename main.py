"""Entry point for the GTA VI launch-day Twitch CCU forecast.

The grader runs this file as `python3 -P -s /bot/main.py` with
PYTHONHASHSEED=0. The -P flag keeps this file's own directory off
sys.path, so the first thing this module does is put that directory
back at the front of the path; only then can it load the forecast
model that lives in the src/ package next to it.

For every `<id>.in` case file found in the input directory (in name
order) it writes one `<id>.out` file to the output directory: a JSON
object with exactly one key, "hours", holding 23 rows for hours 2..24,
each row five quantiles at the levels 0.05, 0.25, 0.5, 0.75 and 0.95.
The input and output directories come from MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR when set, and default to /input and /output.

A case must never take the entry down with it: each case is handled
inside its own guard, and any failure while forecasting a case falls
back to the persistence baseline built from that case's hour-1 print
(or from 1 viewer when even that cannot be read), so the output file
is still written and still valid. The program uses only the standard
library, touches no network, and keeps no state between cases, so two
runs over the same input produce byte-identical output.
"""

import json
import os
import sys
from pathlib import Path

# -P keeps this directory off sys.path: restore it first so the src
# package sitting beside this file can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import model  # noqa: E402  (import needs the path fix above)

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def _hour1_print(case):
    """Hour-1 viewers from a parsed case, or None when unreadable."""
    try:
        for row in case.get("known") or []:
            if isinstance(row, dict) and row.get("hour") == 1:
                value = row.get("viewers")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if value > 0:
                        return float(value)
    except (AttributeError, TypeError):
        return None
    return None


def _forecast_rows(case):
    """Forecast rows for one case, always valid, never raising."""
    try:
        rows = model.forecast_case(case)
    except Exception:
        rows = None
    if not isinstance(rows, dict) or len(rows) != 23:
        anchor = _hour1_print(case)
        rows = model.baseline_forecast(anchor if anchor else 1.0)
    hours = []
    for hour in range(2, 25):
        quantiles = rows.get(hour)
        if (not isinstance(quantiles, (list, tuple))
                or len(quantiles) != 5):
            quantiles = model.baseline_forecast(
                _hour1_print(case) or 1.0)[hour]
        hours.append({"hour": hour, "q": [float(q) for q in quantiles]})
    return {"hours": hours}


def _process_case(path, output_dir):
    """Read one case file and write its forecast file."""
    case_id = path.stem
    try:
        with open(path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        if not isinstance(case, dict):
            raise ValueError("case is not a JSON object")
        if isinstance(case.get("id"), str) and case["id"]:
            case_id = case["id"]
    except Exception:
        # The case file itself is unreadable: fall through with an
        # empty case, so the run still emits a valid baseline
        # forecast under the file's own stem instead of leaving
        # the output missing.
        case = {}
    payload = _forecast_rows(case)
    text = json.dumps(payload, separators=(",", ":")) + "\n"
    target = output_dir / (case_id + ".out")
    tmp = output_dir / ("." + case_id + ".out.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, target)


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_files = sorted(INPUT_DIR.glob("*.in"), key=lambda p: p.name)
    for path in case_files:
        try:
            _process_case(path, OUTPUT_DIR)
        except Exception:
            # Last-resort guard: one bad case must never abort the run
            # or prevent the remaining cases from being forecast.
            try:
                anchor_rows = model.baseline_forecast(1.0)
                payload = {"hours": [
                    {"hour": h, "q": [float(q) for q in anchor_rows[h]]}
                    for h in range(2, 25)]}
                text = json.dumps(payload, separators=(",", ":")) + "\n"
                target = OUTPUT_DIR / (path.stem + ".out")
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write(text)
            except Exception:
                pass


if __name__ == "__main__":
    main()
