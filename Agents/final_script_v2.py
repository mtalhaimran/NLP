import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Any
import sys

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "Agents" / "Crawler"))
sys.path.append(str(BASE_DIR / "Agents" / "AI Service"))

from ai_lead_generation_agent import search_for_urls, load_page_text
from agency import B2BAgency, Project
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    PageBreak,
    ListFlowable,
    Preformatted,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors


# ---- Lead Scoring ----
def score_leads(urls: List[str]) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    for url in urls:
        try:
            text = load_page_text(url)
            ans_match = re.search(r"(\d+(?:,\d+)*)\s+answers", text, re.I)
            if not ans_match:
                ans_match = re.search(r"(\d+(?:,\d+)*)\s+answer", text, re.I)
            view_match = re.search(r"(\d+(?:,\d+)*)\s+views", text, re.I)
            answers = int(ans_match.group(1).replace(",", "")) if ans_match else 0
            views = int(view_match.group(1).replace(",", "")) if view_match else 0
            score = answers * 0.7 + views * 0.3
        except Exception:
            answers = views = score = 0.0
        scores[url] = {"answers": answers, "views": views, "score": score}
    return scores


# ---- PDF Generation ----
def draw_border(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.lightgrey)
    x, y, width, height = doc.leftMargin, doc.bottomMargin, doc.width, doc.height
    canvas.rect(x, y, width, height, stroke=1, fill=0)
    canvas.restoreState()


def export_pdf(results: Dict[str, Any], metrics: Dict[str, float], url: str, out: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("DocTitle", parent=styles["Heading1"], fontSize=28, leading=32, alignment=1, spaceAfter=24))
    styles.add(ParagraphStyle("Subtitle", parent=styles["Heading2"], fontSize=16, leading=20, alignment=1, textColor=colors.grey, spaceAfter=48))
    styles.add(ParagraphStyle("Section", parent=styles["Heading2"], fontSize=18, leading=22, textColor=colors.darkblue, spaceBefore=12, spaceAfter=8))
    styles.add(ParagraphStyle("SubHeading", parent=styles["Heading3"], fontSize=14, leading=18, spaceBefore=6, spaceAfter=4))
    styles["Normal"].fontSize = 11
    styles["Normal"].leading = 14

    def to_flowables(data: Any) -> List[Any]:
        flows: List[Any] = []
        if isinstance(data, dict):
            for k, v in data.items():
                flows.append(Paragraph(str(k).replace("_", " ").title(), styles["SubHeading"]))
                flows.extend(to_flowables(v))
        elif isinstance(data, list):
            if all(isinstance(i, (str, int, float)) for i in data):
                items = [Paragraph(str(i), styles["Normal"]) for i in data]
                flows.append(ListFlowable(items, bulletType="bullet", leftIndent=20, spaceBefore=2, spaceAfter=6))
            else:
                for item in data:
                    flows.extend(to_flowables(item))
        else:
            flows.append(Paragraph(str(data), styles["Normal"]))
        return flows

    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=1 * inch, bottomMargin=1 * inch)

    story: List[Any] = [
        Paragraph(url, styles["DocTitle"]),
        Paragraph(f"Answers: {metrics['answers']} | Views: {metrics['views']} | Score: {metrics['score']:.2f}", styles["Subtitle"]),
        PageBreak(),
    ]

    for role, content in results.items():
        story.append(Paragraph(f"{role} - Summary", styles["Section"]))
        snippet = json.dumps(content, indent=2) if isinstance(content, (dict, list)) else str(content)
        story.append(Preformatted(snippet[:800], styles["Normal"]))
        story.append(PageBreak())
        story.append(Paragraph(f"{role} - Details", styles["Section"]))
        story.extend(to_flowables(content))
        story.append(PageBreak())

    if story and isinstance(story[-1], PageBreak):
        story.pop()

    doc.build(story, onFirstPage=draw_border, onLaterPages=draw_border)


# ---- Agent Execution ----
async def run_agents(text: str) -> Dict[str, Any]:
    agency = B2BAgency()
    project = Project(name="Lead Analysis", description=text, project_type="Lead", budget="n/a", timeline="n/a", priority="High")
    pj = project.model_dump_json()

    with st.spinner("Running CEO..."):
        await agency._run_agent("CEO", pj)

    cto_task = asyncio.create_task(agency._run_agent("CTO", pj, ["CEO"]))

    async def rest_flow() -> None:
        await agency.events["CTO"].wait()
        with st.spinner("Running Product Manager..."):
            await agency._run_agent("PM", pj, ["CTO"])
        with st.spinner("Running Marketing Manager and Developer..."):
            dev = asyncio.create_task(agency._run_agent("DEV", pj, ["CTO", "PM"]))
            mkt = asyncio.create_task(agency._run_agent("MARKETING", pj, ["PM"]))
            await asyncio.gather(dev, mkt)
        with st.spinner("Running Client Success..."):
            await agency._run_agent("CLIENT", pj, ["DEV"])

    await asyncio.gather(cto_task, rest_flow())
    return agency.results


# ---- Streamlit UI ----
st.title("Auto Lead")

with st.sidebar:
    num_links = st.number_input("Number of links", min_value=1, max_value=10, value=3)
    company_desc = st.text_input("Lead Search Query", "AI chatbots for e-commerce")

if st.button("Find Leads"):
    with st.spinner("Searching Google..."):
        urls = search_for_urls(company_desc, num_links)
    scores = score_leads(urls)
    if scores:
        top_url, metrics = max(scores.items(), key=lambda x: x[1]["score"])
        st.session_state["top_url"] = top_url
        st.session_state["metrics"] = metrics
        st.write("Top lead:", top_url)
        st.write(f"Score: {metrics['score']:.2f} (Answers {metrics['answers']}, Views {metrics['views']})")
    else:
        st.warning("No leads found")

if "top_url" in st.session_state:
    if st.button("Analyze Lead"):
        page_text = load_page_text(st.session_state["top_url"])
        results = asyncio.run(run_agents(page_text))
        pdf_file = Path("lead_report.pdf")
        export_pdf(results, st.session_state["metrics"], st.session_state["top_url"], pdf_file)
        st.download_button("Download Report", data=open(pdf_file, "rb").read(), file_name="lead_report.pdf")
