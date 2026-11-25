"""Deliver the checked single-file report in this campaign and open the actual copy offline."""
import json
import os
from datetime import datetime
from tempfile import NamedTemporaryFile

from playwright.sync_api import sync_playwright

from lock_from_frontier import ROOT, REVIEW, read, digest


def main():
    earlier = ROOT.parent / "overnight-20261122/review"
    packed = earlier / "report-src/dist/index.html"
    render = read(earlier / "render_check.json")
    assert digest(packed) == render["html_sha256"] and render["report_status"] == "complete"
    destination = REVIEW / "REPORT.html"
    with NamedTemporaryFile(dir=REVIEW, prefix="report-delivery-", suffix=".tmp", delete=False) as temp:
        temp.write(packed.read_bytes())
    os.replace(temp.name, destination)
    assert digest(destination) == render["html_sha256"]
    checked = {}
    with sync_playwright() as shell:
        browser = shell.chromium.launch()
        for width in (390, 1280):
            context = browser.new_context(viewport={"width":width,"height":900}, locale="zh-CN", offline=True)
            page = context.new_page()
            errors, external = [], []
            page.on("pageerror", lambda e:errors.append(str(e)))
            page.on("request", lambda r:external.append(r.url) if r.url.startswith(("http://","https://")) else None)
            page.goto(destination.as_uri())
            table = page.locator("#study-section-all8-final-cross-source").get_by_role("table")
            table.wait_for(state="visible", timeout=15000)
            assert table.get_by_role("row").count() == 13
            assert page.locator("#study-section-all8-partitions").get_by_role("table").get_by_role("row").count()==13
            assert page.locator("#study-section-all8-frozen-external-models").get_by_role("table").get_by_role("row").count()==13
            assert page.locator("#study-section-all8-self-encoder-clock").get_by_role("table").get_by_role("row").count()==16
            assert len(page.locator("#study-section-all8-supervision table tbody tr").all())==5
            page.locator("[data-final-gain-chart]").scroll_into_view_if_needed()
            bar = page.locator("[data-final-gain-chart] svg .recharts-bar-rectangle .recharts-rectangle")
            bar.first.wait_for(state="visible")
            assert bar.count()==12 and not errors and not external
            assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
            checked[str(width)]={"offline":True, "network_requests":len(external), "page_errors":errors,
                                 "tables":page.get_by_role("table").count(), "gain_chart_domains":bar.count()}
            context.close()
        browser.close()
    result={"verified_at":datetime.now().astimezone().isoformat(), "source_render_sha256":digest(earlier/"render_check.json"),
            "html_sha256":digest(destination), "bytes":destination.stat().st_size,"viewports":checked}
    (REVIEW/"published-report-qa.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(result["bytes"],result["html_sha256"],result["viewports"])


if __name__=="__main__":
    main()
