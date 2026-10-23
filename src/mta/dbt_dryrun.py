"""Dry-run every compiled dbt model and test and print bytes processed.

Run after `dbt compile`. Models whose upstream tables don't exist yet (first run) are
reported as skipped. Exits non-zero if any query exceeds the per-query cap.
"""

from __future__ import annotations

import sys

from google.api_core.exceptions import BadRequest, NotFound

from mta import bq

COMPILED = bq.ROOT / "dbt" / "target" / "compiled" / "ga4_mta"


def main() -> int:
    files = sorted(COMPILED.rglob("*.sql"))
    if not files:
        print("No compiled SQL found; run `dbt compile` first.")
        return 1
    total, over, skipped = 0, [], 0
    for f in files:
        rel = f.relative_to(COMPILED)
        try:
            n = bq.dry_run(f.read_text())
        except (NotFound, BadRequest) as e:
            if "Not found" in str(e):
                skipped += 1
                continue
            raise
        total += n
        flag = "  <-- OVER CAP" if n > bq.MAX_BYTES else ""
        print(f"  {n / 1e6:>8,.1f}MB  {rel}{flag}")
        if flag:
            over.append(rel)
    if skipped:
        print(f"  ({skipped} queries skipped: their upstream tables aren't built yet; run `make dbt` first)")
    print(f"Total if everything ran once: {total / 1e9:,.2f} GB (free tier: 1,000 GB / month)")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
