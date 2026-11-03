"""Vector PDF of an SVG figure, page size = the SVG's mm size, via headless
Chromium (same engine and fonts as render.py). Text stays live.

    python scripts/render_pdf.py fig.svg fig.pdf
"""

import re
import sys

from playwright.sync_api import sync_playwright

svg, pdf = sys.argv[1], sys.argv[2]
content = open(svg).read()
m = re.search(r'width="([0-9.]+)mm" height="([0-9.]+)mm"', content)
w_mm, h_mm = float(m.group(1)), float(m.group(2))
content = content.replace(m.group(0), 'width="100%" height="100%"', 1)
html = (
    "<html><head><style>@page{margin:0}html,body{margin:0;"
    f"width:{w_mm}mm;height:{h_mm}mm}}</style></head><body>"
    + content
    + "</body></html>"
)
with sync_playwright() as p:
    b = p.chromium.launch(args=["--disable-gpu", "--no-sandbox"])
    pg = b.new_page()
    pg.set_content(html)
    pg.wait_for_timeout(300)
    pg.pdf(
        path=pdf,
        width=f"{w_mm}mm",
        height=f"{h_mm}mm",
        print_background=True,
        margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
    )
    b.close()
print("rendered", pdf, f"{w_mm} x {h_mm} mm")
