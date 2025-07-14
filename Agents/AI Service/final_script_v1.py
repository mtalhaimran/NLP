import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st
from bs4 import BeautifulSoup
from googlesearch import search
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
)

REPO_DIR = Path(__file__).resolve().parent
MIN_LEAD_SCORE = 50.0
sys.path.append(str(REPO_DIR / "Agents" / "AI Service"))

from agency import B2BAgency, Project, draw_border

# ─── CRAWLER UTILITIES ──────────────────────────────────────────────────────────


def search_google_quora(company_description: str, num_links: int) -> List[str]:
    query = f"site:quora.com {company_description}"
    urls: List[str] = []
    for url in search(query, num_results=num_links * 2, lang="en"):
        if "quora.com" in url:
            urls.append(url)
        if len(urls) >= num_links:
            break
    return urls


def search_for_urls(company_description: str, num_links: int) -> List[str]:
    return search_google_quora(company_description, num_links)


def load_page_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def parse_metrics(text: str) -> tuple[int, int]:
    a = re.search(r"(\d[\d,]*)\\s+Answers?", text, re.I)
    v = re.search(r"(\d[\d,]*)\\s+Views?", text, re.I)
    answers = int(a.group(1).replace(",", "")) if a else 0
    views = int(v.group(1).replace(",", "")) if v else 0
    return answers, views


def score_leads(urls: List[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for u in urls:
        try:
            t = load_page_text(u)
            ans, views = parse_metrics(t)
            scores[u] = ans * 0.7 + views * 0.3
        except Exception:
            scores[u] = 0.0
    return scores


# ─── AGENCY SUBCLASS WITH PDF EXPORT ────────────────────────────────────────────
class LeadAgency(B2BAgency):
    def export_pdf(self, lead_url: str, metrics: Dict[str, int], out_path: Path) -> None:
        def to_flowables(data: Any) -> List[Any]:
            flows: List[Any] = []
            if isinstance(data, dict):
                for k, v in data.items():
                    flows.append(
                        Paragraph(str(k).replace("_", " ").title(), styles["SubHeading"])
                    )
                    flows.extend(to_flowables(v))
            elif isinstance(data, list):
                if all(isinstance(i, (str, int, float)) for i in data):
                    items = [
                        Paragraph(str(i), styles["NormalLeft"]) for i in data
                    ]
                    flows.append(
                        ListFlowable(
                            items,
                            bulletType="bullet",
                            leftIndent=0,
                            rightIndent=0,
                            spaceBefore=2,
                            spaceAfter=6,
                        )
                    )
                else:
                    for item in data:
                        flows.extend(to_flowables(item))
            elif isinstance(data, (str, int, float)):
                flows.append(Paragraph(str(data), styles["NormalLeft"]))
            else:
                flows.append(
                    Preformatted(json.dumps(data, indent=2), styles["NormalLeft"])
                )
            return flows

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=letter,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=1 * inch,
            bottomMargin=1 * inch,
        )

        global styles
        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                "DocTitle",
                parent=styles["Heading1"],
                fontSize=28,
                leading=32,
                alignment=1,
                spaceAfter=24,
            )
        )
        styles.add(
            ParagraphStyle(
                "Subtitle",
                parent=styles["Heading2"],
                fontSize=16,
                leading=20,
                alignment=1,
                textColor=colors.grey,
                spaceAfter=48,
            )
        )
        styles.add(
            ParagraphStyle(
                "Section",
                parent=styles["Heading2"],
                fontSize=18,
                leading=22,
                textColor=colors.darkblue,
                spaceBefore=12,
                spaceAfter=8,
            )
        )
        styles.add(
            ParagraphStyle(
                "SubHeading",
                parent=styles["Heading3"],
                fontSize=14,
                leading=18,
                spaceBefore=6,
                spaceAfter=4,
            )
        )
        styles["Normal"].fontSize = 11
        styles["Normal"].leading = 14
        styles.add(ParagraphStyle("NormalLeft", parent=styles["Normal"], alignment=0))

        story: List[Any] = [
            Paragraph(lead_url, styles["DocTitle"]),
            Paragraph(
                f"Answers: {metrics.get('answers',0)} | Views: {metrics.get('views',0)}",
                styles["Subtitle"],
            ),
            PageBreak(),
        ]

        for role, content in self.results.items():
            summary = json.dumps(content, indent=2)
            story.append(Paragraph(f"{role} Summary", styles["Section"]))
            story.append(Preformatted(summary, styles["Normal"]))
            story.append(PageBreak())

            story.append(Paragraph(f"{role} Details", styles["Section"]))
            story.extend(to_flowables(content))
            story.append(PageBreak())

        if story and isinstance(story[-1], PageBreak):
            story.pop()

        doc.build(story, onFirstPage=draw_border, onLaterPages=draw_border)


# ─── ASYNC PIPELINE ORCHESTRATION ──────────────────────────────────────────────
async def analyze_text(
    text: str,
    name: str,
    project_type: str,
    budget: str,
    timeline: str,
    priority: str,
) -> LeadAgency:
    agency = LeadAgency()
    project = Project(
        name=name,
        description=text,
        project_type=project_type,
        budget=budget,
        timeline=timeline,
        priority=priority,
    )
    pj = project.model_dump_json()

    async def run(role: str, deps: List[str] | None = None):
        if deps:
            await asyncio.gather(*(agency.events[d].wait() for d in deps))
        with st.spinner(f"Running {role}…"):
            res = await agency.agent_map[role].run(pj, agency.context, agency.timeout)
        agency.context[role] = res
        agency.results[role] = res
        agency.events[role].set()

    await run("CEO")
    cto_task = asyncio.create_task(run("CTO", ["CEO"]))

    async def rest():
        await agency.events["CTO"].wait()
        await run("PM", ["CTO"])
        dev_task = asyncio.create_task(run("DEV", ["CTO", "PM"]))
        marketing_task = asyncio.create_task(run("MARKETING", ["PM"]))
        await asyncio.gather(dev_task, marketing_task)
        await run("CLIENT", ["DEV"])

    await asyncio.gather(cto_task, rest())
    return agency


# ─── STREAMLIT UI ──────────────────────────────────────────────────────────────
st.title("🎯 Lead Crawler & AI Report Generator")

with st.sidebar:
    st.header("Search Settings")
    num_links = st.number_input("Number of links", min_value=1, max_value=10, value=3)
    st.header("Proposal Options")
    project_name = st.text_input("Project name", value="Quora Lead")
    project_type = st.text_input("Project type", value="Lead")
    budget = st.text_input("Budget", value="N/A")
    timeline = st.text_input("Timeline", value="N/A")
    priority = st.selectbox("Priority", ["High", "Medium", "Low"], index=0)
    pdf_filename = st.text_input("Output PDF filename", value="lead_report.pdf")
    if st.button("Reset"):
        st.session_state.clear()
        st.experimental_rerun()

query = st.text_input(
    "Describe your target leads",
    placeholder="e.g., businesses needing AI video editing software",
)

if st.button("Find Leads"):
    if not query:
        st.error("Please provide a description.")
    else:
        with st.spinner("Searching for relevant URLs…"):
            urls = search_for_urls(query, num_links)
        if not urls:
            st.warning("No URLs found.")
        else:
            with st.spinner("Scoring leads…"):
                scores = score_leads(urls)
            top_url = max(scores, key=scores.get)
            top_score = scores[top_url]
            st.session_state["top_url"] = top_url
            st.session_state["top_score"] = top_score
            st.write("Top lead:", top_url)
            st.write("Lead score:", f"{top_score:.2f}")

if "top_url" in st.session_state:
    top_score = st.session_state.get("top_score", 0.0)
    lead_url = st.session_state["top_url"]

    def run_analysis() -> None:
        page_text = load_page_text(lead_url)
        answers, views = parse_metrics(page_text)
        agency = asyncio.run(
            analyze_text(
                page_text,
                project_name,
                project_type,
                budget,
                timeline,
                priority,
            )
        )
        pdf_file = REPO_DIR / pdf_filename
        agency.export_pdf(lead_url, {"answers": answers, "views": views}, pdf_file)
        with open(pdf_file, "rb") as f:
            st.download_button(
                "Download Report",
                data=f.read(),
                file_name=pdf_filename,
            )

    if top_score < MIN_LEAD_SCORE:
        st.warning(
            f"Lead score {top_score:.2f} is below the minimum of {MIN_LEAD_SCORE}. "
            "Refine your search or adjust options."
        )
        if st.button("Analyze Anyway"):
            run_analysis()
    else:
        if st.button("Analyze Lead"):
            run_analysis()
