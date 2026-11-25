"""Check the actual offline report in desktop and mobile Chromium viewports."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright


HERE = Path(__file__).resolve().parent
APP = HERE / "report-src"
data = json.loads((APP / "src/data.json").read_text(encoding="utf-8"))
story = data["report"]["review"]
metrics = data["queries"]["historical"]["rows"]
counts = data["queries"]["rounds"]["rows"]
table_count = 2 + sum(bool(section.get("columns")) for section in story["extraSections"])
first_ids = ["all8-final-cross-source","all8-final-vqf-controls","all8-final-self","all8-final","all8-partitions","all8-accounting"] if "all8-final-cross-source" in data["queries"] else []
indexed = {s["id"]:s for s in story["extraSections"]}
narratives = [story["byline"], story["summary"], *(indexed[key]["body"] for key in first_ids),
              story["counts"], story["comparison"], *(s["body"] for s in story["extraSections"] if s["id"] not in first_ids)]
result = {}
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    for width in (1280, 390):
        page = browser.new_page(viewport={"width": width, "height": 900}, locale="zh-CN")
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        url = (APP / "dist/index.html").as_uri()
        has_split = "dataset_partitions" in data["queries"]
        page.goto(url + ("#study-section-dataset-split" if has_split else ""))
        page.wait_for_selector("table", timeout=15000)
        if has_split:
            page.wait_for_function("""() => {
                const heading = document.querySelector("#study-section-dataset-split h2");
                const top = heading?.getBoundingClientRect().top;
                return top >= 0 && top < 250;
            }""", timeout=5000)
        assert page.locator("h1").inner_text() == data["title"]
        rendered = page.locator("article.report-content .rich-narrative-content").all_inner_texts()
        for position, (source, visible) in enumerate(zip(narratives, rendered, strict=True)):
            plain = re.sub(r"(?m)^#{1,6} +", "", source).replace("**", "").replace("`", "")
            assert "".join(plain.split()) == "".join(visible.split()), (width, "narrative", position)
        tables = page.get_by_role("table")
        assert tables.count() == table_count
        count_table=page.locator("#study-section-round-counts").get_by_role("table")
        historical_table=page.locator("#study-section-historical-baselines").get_by_role("table")
        count_text = count_table.inner_text()
        for row in counts:
            assert row["group"] in count_text
            line = count_table.get_by_role("row").filter(has=page.get_by_role("cell", name=row["group"], exact=True))
            assert line.count() == 1
            expected = [row["group"], f'{row["rounds"]:,d}', row["median_time"], row["best"], row["round_id"]]
            actual = [cell.strip() for cell in line.get_by_role("cell").all_inner_texts()]
            assert actual == expected, (width, row["group"], expected, actual)
        for row in metrics:
            line = historical_table.get_by_role("row").filter(has_text=row["method"])
            assert line.count() == 1
            expected = [row["method"]] + [f"{row[d]:.3f}°" for d in ("R6", "BROAD", "Sassari")]
            assert [cell.strip() for cell in line.get_by_role("cell").all_inner_texts()] == expected
        extra_tables = [section for section in story["extraSections"] if section.get("columns")]
        for section in extra_tables:
            table = page.locator("#study-section-"+section["id"]).get_by_role("table")
            query_rows = data["queries"][section["queryId"]]["rows"]
            body_rows = table.get_by_role("row")
            assert body_rows.count() == len(query_rows) + 1
            for i, row in enumerate(query_rows, start=1):
                expected = [str(row[column["field"]]) for column in section["columns"]]
                rendered_cells=[s.strip() for s in body_rows.nth(i).get_by_role("cell").all_inner_texts()]
                assert rendered_cells == expected, (width, section["id"], i, expected, rendered_cells)
        assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        links = page.get_by_role("navigation", name="报告目录").get_by_role("link")
        available = {"round-counts", "historical-baselines", *(section["id"] for section in story["extraSections"])}
        destinations = [key for key in ("all8-final", "all8-final-cross-source", "all8-final-vqf-controls", "all8-supplemental", "all8-final-penn", "all8-final-latency", "all8-development", "all8-partitions", "all8-guided", "transfer-mamba", "dataset-split", "round-counts", "historical-baselines", "self-records", "locked-components", "untuned-external", "timing") if key in available]
        assert links.count() == len(destinations)
        for position, key in enumerate(destinations):
            link = links.nth(position)
            target = "#study-section-" + key
            assert link.get_attribute("href") == target
            link.click()
            page.wait_for_function("target => location.hash === target", arg=target)
            header = page.locator(target).get_by_role("heading").first
            assert header.is_visible()
            assert 0 <= header.bounding_box()["y"] < 250, (width, key)
        chart_marks=0
        if "all8-final-cross-source" in data["queries"]:
            chart=page.locator('[data-final-gain-chart]')
            chart.scroll_into_view_if_needed()
            assert chart.count()==1 and chart.get_by_role('alert').count()==0
            plotted=[r for r in data['queries']['all8-final-cross-source']['rows'] if r.get('gain_fraction') is not None]
            expected_marks=len(plotted)
            marks=chart.locator('svg .recharts-bar-rectangle .recharts-rectangle, svg [data-chart-mark]')
            marks.first.wait_for(state='visible',timeout=15000)
            chart_marks=marks.count()
            assert chart_marks==expected_marks,(width,chart_marks,expected_marks)
            svg_text=chart.locator('svg text').evaluate_all("nodes => nodes.map(n => n.textContent)")
            widths=marks.evaluate_all("nodes => nodes.map(n => Number(n.getAttribute('width')))")
            for i,row in enumerate(plotted):
                label=row['c0'].replace('_',' ')
                fraction=row['gain_fraction']
                value=f"{100*fraction:.1f}".removesuffix('.0')+'%'
                assert label in svg_text[i] and svg_text[i+chart_marks]==value,(width,label,value,svg_text)
                assert widths[i]*fraction>0,(width,label,widths[i],fraction)
            pixels_per_unit=[n/r['gain_fraction'] for n,r in zip(widths,plotted)]
            assert max(pixels_per_unit)/min(pixels_per_unit)<1.001
        assert not errors, errors
        result[str(width)] = {"tables": table_count, "verified_narratives": len(rendered), "count_rows": len(counts), "paired_comparison_rows": len(metrics),
                              "verified_section_links": len(destinations), "gain_chart_marks":chart_marks, "document_overflow": False, "page_errors": errors}
        page.close()
    browser.close()
review = {"reviewed_utc": datetime.now(timezone.utc).isoformat(), "report_status": data["buildStatus"],
          "html_sha256":hashlib.sha256((APP / "dist/index.html").read_bytes()).hexdigest(),
          "source_generated_at": data["generatedAt"], "viewports_css_px": result}
(HERE / "render_check.json").write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(review, ensure_ascii=False))
