"""Capture dashboard screenshots for the README (needs the dashboard running on --port)."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
PAGES = {"story": "", "attribution": "attribution", "budget": "budget", "funnel": "funnel", "data": "data"}


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
            for _ in range(30):  # first load computes the cached data; wait for charts to appear
                if page.locator(".js-plotly-plot").count():
                    break
                page.wait_for_timeout(1000)
            page.wait_for_timeout(2500)  # let plotly finish drawing
            errors = page.locator("[data-testid='stException']").count()
            if errors or page.get_by_text("Script execution error").count():
                raise SystemExit(f"Page '{name}' rendered a Streamlit error")
            page.screenshot(path=str(out / f"{name}.png"), full_page=True)
            print(f"  docs/screenshots/{name}.png")
        browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    main(ap.parse_args().port)
