"""Capture dashboard screenshots for the README (needs the dashboard running on --port)."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
PAGES = {"overview": "", "funnel": "funnel", "journeys": "journeys", "attribution": "attribution",
         "intervals": "intervals", "budget": "budget"}


def main(port: int) -> None:
    out = ROOT / "docs" / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
        for name, path in PAGES.items():
            page.goto(f"http://localhost:{port}/{path}")
            page.wait_for_selector("[data-testid='stMainBlockContainer']", timeout=60000)
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(4000)  # let plotly finish drawing
            errors = page.locator("[data-testid='stException']").count()
            if errors:
                raise SystemExit(f"Page '{name}' rendered a Streamlit exception")
            page.screenshot(path=str(out / f"{name}.png"), full_page=True)
            print(f"  docs/screenshots/{name}.png")
        browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    main(ap.parse_args().port)
