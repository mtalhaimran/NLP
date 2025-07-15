import json
from typing import Any, List

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, ListFlowable, Preformatted


def init_styles():
    """Return a ReportLab stylesheet used for proposal PDFs."""
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=28,
            leading=32,
            alignment=1,
            spaceAfter=24,
            fontName="Helvetica-Bold",
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
            fontName="Helvetica-Bold",
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
            fontName="Helvetica-Bold",
        )
    )
    styles["Normal"].fontSize = 11
    styles["Normal"].leading = 14
    styles.add(ParagraphStyle("NormalLeft", parent=styles["Normal"], alignment=0))
    return styles


def to_flowables(data: Any, styles) -> List[Any]:
    """Convert nested data structures into ReportLab flowables."""
    flows: List[Any] = []
    if isinstance(data, dict):
        if set(data.keys()) == {"heading", "content"}:
            flows.append(Paragraph(str(data["heading"]), styles["SubHeading"]))
            flows.extend(to_flowables(data["content"], styles))
        else:
            for k, v in data.items():
                if k == "sections" and isinstance(v, list):
                    for item in v:
                        flows.extend(to_flowables(item, styles))
                    continue
                flows.append(
                    Paragraph(str(k).replace("_", " ").title(), styles["SubHeading"])
                )
                flows.extend(to_flowables(v, styles))
    elif isinstance(data, list):
        if all(isinstance(i, (str, int, float)) for i in data):
            items = [Paragraph(str(i), styles["NormalLeft"]) for i in data]
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
                flows.extend(to_flowables(item, styles))
    elif isinstance(data, (str, int, float)):
        flows.append(Paragraph(str(data), styles["NormalLeft"]))
    else:
        flows.append(
            Preformatted(
                json.dumps(data, indent=2), styles["NormalLeft"], maxLineLength=80
            )
        )
    return flows

