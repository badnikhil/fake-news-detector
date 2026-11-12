# Web UI screenshots

Taken 2026-11-12 in **Firefox 157 (snap, headless)**, driven through geckodriver 0.37.1's WebDriver API, against `MODE=offline` uvicorn on port 8010 (DistilBERT classifier, offline FAISS index). PNGs are quantised to 128 colours; desktop pages are cropped at 3200 px height.

| File | What |
|---|---|
| `01_input_desktop_light.png` | Analyse screen, 1280 px, light theme |
| `02_result_real.png` | Gallery example REAL (R7), corporate-tax headline |
| `03_result_fake.png` | Gallery example FAKE (R2), Phoenix kidnapping headline |
| `04_result_misleading.png` | Gallery example MISLEADING (R4), headline contradicted by its body |
| `05_about_desktop_light.png` | About screen |
| `06_result_fake_dark.png` | FAKE result, dark theme (Theme toggle) |
| `07_result_misleading_phone.png` | MISLEADING result at 375 px width (in a 375 px iframe, because headless Firefox will not shrink the window below 500 px) |

Checks done in the same run: all 5 gallery examples gave their expected verdict (REAL, FAKE, PARTIALLY TRUE, MISLEADING, UNVERIFIABLE; 1 to 5 s each). Download JSON saved `analysis-<request_id>.json`, which parses and contains `verdict`. The About screen renders. At 375 px, `scrollWidth == clientWidth` on the Analyse, Result and About screens, so there is no horizontal scroll.
