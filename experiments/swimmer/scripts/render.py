import re
import sys
from playwright.sync_api import sync_playwright

svg, png = sys.argv[1], sys.argv[2]
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
content = open(svg).read()
vb = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', content)
w, h = int(float(vb.group(1))), int(float(vb.group(2)))
content = re.sub(r'width="[0-9.]+mm" height="[0-9.]+mm"',
                 f'width="{w}" height="{h}"', content, count=1)
content = re.sub(r'width="100%" height="100%"', f'width="{w}" height="{h}"', content, count=1)
html = ("<html><body style='margin:0;background:white'>" + content +
        "</body></html>")
with sync_playwright() as p:
    b = p.chromium.launch(args=["--disable-gpu", "--no-sandbox"])
    pg = b.new_page(viewport={"width": w, "height": h},
                    device_scale_factor=scale)
    pg.set_content(html)
    pg.wait_for_timeout(300)
    pg.screenshot(path=png, clip={"x": 0, "y": 0, "width": w, "height": h},
                  timeout=60000)
    b.close()
print("rendered", w, h)
