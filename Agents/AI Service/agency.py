#!/usr/bin/env python3
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

import typer
from pydantic import BaseModel, Field, ValidationError
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Preformatted,
    ListFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)

def system_prompt(template: str) -> SystemMessagePromptTemplate:
    """Compatibility wrapper for SystemMessagePromptTemplate.from_template."""
    if hasattr(SystemMessagePromptTemplate, "from_template"):
        return SystemMessagePromptTemplate.from_template(template)
    return SystemMessagePromptTemplate(prompt=PromptTemplate.from_template(template))


def human_prompt(template: str) -> HumanMessagePromptTemplate:
    """Compatibility wrapper for HumanMessagePromptTemplate.from_template."""
    if hasattr(HumanMessagePromptTemplate, "from_template"):
        return HumanMessagePromptTemplate.from_template(template)
    return HumanMessagePromptTemplate(prompt=PromptTemplate.from_template(template))
from langchain_ollama import ChatOllama

# ─── CONFIGURATION ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_TAGS = {role: "mistral:7b-instruct" for role in ["CEO", "CTO", "PM", "DEV", "MARKETING", "CLIENT"]}

# Default timeout (in seconds) for each agent run. The models can be quite
# slow to respond, especially on lower-spec hardware. 60 seconds proved too
# short in practice, so we use a more generous default and expose it via the
# CLI for user customisation.
DEFAULT_TIMEOUT: float = 180.0

# ─── DATA SCHEMA ───────────────────────────────────────────────────────
class Project(BaseModel):
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Full project description")
    project_type: str = Field(..., description="Project type")
    budget: str = Field(..., description="Budget range, e.g. $25k-$50k")
    timeline: str = Field(..., description="Expected timeline, e.g. 3-4 months")
    priority: str = Field(..., description="High | Medium | Low")

# ─── PDF BORDER ────────────────────────────────────────────────────────
def draw_border(canvas, doc):
    """Draw a light grey border respecting the document margins."""
    canvas.saveState()
    canvas.setStrokeColor(colors.lightgrey)
    x = doc.leftMargin
    y = doc.bottomMargin
    width = doc.width
    height = doc.height
    canvas.rect(x, y, width, height, stroke=1, fill=0)
    canvas.restoreState()

# ─── PROMPT TEMPLATES ──────────────────────────────────────────────────
# 1. Chief Strategy Officer: Go/No-Go analysis
CEO_PROMPT = ChatPromptTemplate.from_messages([
    system_prompt(
        '''
You are the Chief Strategy Officer. Review the provided project data and perform a Go/No-Go analysis.
Generate a structured JSON response matching this schema. Optionally include a "title" summarizing your analysis:
{{
  "decision": "<GO or NO_GO>",
  "key_risks": ["risk1", "risk2", ...],
  "opportunities": ["op1", "op2", ...],
  "recommendations": [
    {{"title": "<Recommendation title>", "detail": "<Recommendation detail>"}},
    ...
  ]
}}
Start your response with "{{" and return only valid JSON matching this schema.
No additional commentary.
'''    ),
    human_prompt(
        '''
Project Data:
```json
{project}
```
Respond strictly in the JSON format specified.
'''    ),
])

# 2. Chief Technology Officer: System architecture proposal
CTO_PROMPT = ChatPromptTemplate.from_messages([
    system_prompt(
        '''
You are the CTO. Using the project data and the CEO's analysis, propose a system architecture and technology stack.
Organize your output organically: start with an overview, then list components, and conclude with a scalability strategy. Optionally include a "title" summarizing the proposal.
Include one JSON object with keys:
- "architecture": brief summary
 - "components": list of {{"name":..., "purpose":...}}
- "scalability_plan": description
Respond with a bulleted summary plus the JSON block.'''    ),
    human_prompt(
        '''
Project Data:
```json
{project}
```
CEO Analysis:
```json
{CEO}
```
'''
    ),
])

# 3. Product Manager: Phased roadmap
PM_PROMPT = ChatPromptTemplate.from_messages([
    system_prompt(
        '''
You are the Product Manager. Create a three-phase roadmap. Name each phase meaningfully (e.g., "MVP", "Growth", "Scale") and under each phase include bullet points for objectives and deliverables. Optionally add a "title" summarizing this roadmap.
Respond only with valid JSON using this structure:
{{
  "<PhaseName>": {{
    "objectives": ["objective1", ...],
    "deliverables": ["deliverable1", ...]
  }},
  ...
}}
Start your reply with "{{" and ensure it is valid JSON.'''    ),
    human_prompt(
        '''
Project Data:
```json
{project}
```
CTO Specification:
```json
{CTO}
```
'''
    ),
])

# 4. Lead Developer: Task and CI/CD plan
DEV_PROMPT = ChatPromptTemplate.from_messages([
    system_prompt(
        '''
You are the Lead Developer. For each roadmap phase, generate a JSON section containing an optional "title" plus:
- "tasks": list of task descriptions
- "ci_cd": object with fields "tool" and "pipeline_overview"
Let the model assign its own phase labels based on the roadmap headings. Provide valid JSON only.'''    ),
    human_prompt(
        '''
Roadmap:
```markdown
{PM}
```
'''
    ),
])

# 5. Marketing Manager: Go-to-market plan
MARKETING_PROMPT = ChatPromptTemplate.from_messages([
    system_prompt(
        '''
You are the Marketing Manager. Based on the roadmap, craft a concise go-to-market plan. Optionally include a "title" summarizing this section.
Respond only with JSON having the keys:
{{
  "target_audience": ["audience1", ...],
  "channels": ["channel1", ...],
  "messaging": ["theme1", ...]
}}
Start your reply with "{{" and ensure it is valid JSON.'''    ),
    human_prompt(
        '''
Project Data:
```json
{project}
```
Roadmap:
```markdown
{PM}
```
'''
    ),
])

# 6. Client Success Manager: Onboarding and retention
CLIENT_PROMPT = ChatPromptTemplate.from_messages([
    system_prompt(
        '''
You are the Client Success Manager. From the implementation details, draft JSON with an optional "title" and keys:
 - "onboarding_process": array of {{"step":..., "description":...}}
- "retention_strategy": array of strategies
- "feedback_loop": array of feedback mechanisms
Allow the model to name each section. Provide JSON only.'''    ),
    human_prompt(
        '''
Implementation Details:
```json
{DEV}
```
'''
    ),
])

# ─── AGENT & PIPELINE ─────────────────────────────────────────────────
class Agent:
    """Wraps a role-based LLM agent with prompt and model."""
    def __init__(self, role: str, prompt: ChatPromptTemplate):
        self.role = role
        self.prompt = prompt
        model_name = MODEL_TAGS[role]
        logger.info(f"Loading {role} model: {model_name}")
        self.llm = ChatOllama(model=model_name)

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Extract and parse the first JSON object found in text."""
        start = text.find("{")
        if start == -1:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text[start:])
        return obj

    async def run(self, project_json: str, context: Dict[str, Any], timeout: Optional[float] = DEFAULT_TIMEOUT) -> Any:
        """
        Invoke the LLM with formatted messages and return parsed JSON.
        Raises ValueError on invalid JSON, or asyncio.TimeoutError on timeout.
        """
        messages = self.prompt.format_prompt(
            project=project_json,
            CEO=context.get("CEO", ""),
            CTO=context.get("CTO", ""),
            PM=context.get("PM", ""),
            DEV=context.get("DEV", ""),
        ).to_messages()

        try:
            message = await asyncio.wait_for(
                asyncio.to_thread(self.llm.invoke, messages),
                timeout,
            )
            raw = getattr(message, "content", message)
            try:
                parsed = self._parse_json(raw)
            except json.JSONDecodeError:
                logger.warning(
                    f"Non-JSON response from {self.role}, returning raw text"
                )
                parsed = raw
            logger.info(f"{self.role} responded successfully")
            return parsed
        except asyncio.TimeoutError:
            logger.error(f"{self.role} agent timed out after {timeout}s")
            raise


class B2BAgency:
    """Orchestrates a sequence of role-based agents and exports to PDF."""
    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.agents = [
            Agent("CEO", CEO_PROMPT),
            Agent("CTO", CTO_PROMPT),
            Agent("PM", PM_PROMPT),
            Agent("MARKETING", MARKETING_PROMPT),
            Agent("DEV", DEV_PROMPT),
            Agent("CLIENT", CLIENT_PROMPT),
        ]
        self.agent_map = {a.role: a for a in self.agents}
        self.context: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.timeout = timeout
        self.msg_queue: asyncio.Queue = asyncio.Queue()
        self.events: Dict[str, asyncio.Event] = {a.role: asyncio.Event() for a in self.agents}

    async def _run_agent(self, role: str, project_json: str, deps: Optional[List[str]] = None) -> Any:
        if deps:
            await asyncio.gather(*(self.events[d].wait() for d in deps))
        agent = self.agent_map[role]
        logger.info(f"Running {role} agent…")
        res = await agent.run(project_json, self.context, self.timeout)
        self.context[role] = res
        self.results[role] = res
        self.events[role].set()
        await self.msg_queue.put((role, res))
        return res

    async def run_pipeline(self, project: Project) -> Dict[str, Any]:
        pj = project.model_dump_json()

        # CEO runs first
        await self._run_agent("CEO", pj)

        # CTO depends on CEO
        cto_task = asyncio.create_task(self._run_agent("CTO", pj, ["CEO"]))

        async def pm_flow() -> None:
            await self.events["CTO"].wait()
            await self._run_agent("PM", pj, ["CTO"])
            dev_task = asyncio.create_task(self._run_agent("DEV", pj, ["CTO", "PM"]))
            marketing_task = asyncio.create_task(self._run_agent("MARKETING", pj, ["PM"]))
            await asyncio.gather(dev_task, marketing_task)
            await self._run_agent("CLIENT", pj, ["DEV"])

        await asyncio.gather(cto_task, pm_flow())
        return self.results

    def export_pdf(self, project: Project, out_path: Path) -> None:
        """Render proposal results into a formatted PDF."""

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
            elif isinstance(data, (str, int, float)):
                flows.append(Paragraph(str(data), styles["Normal"]))
            else:
                flows.append(Preformatted(json.dumps(data, indent=2), styles["Normal"]))
            return flows

        try:
            doc = SimpleDocTemplate(
                str(out_path),
                pagesize=letter,
                leftMargin=0.75 * inch,
                rightMargin=0.75 * inch,
                topMargin=1 * inch,
                bottomMargin=1 * inch,
            )

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

            story: List[Any] = [
                Paragraph(project.name, styles["DocTitle"]),
                Paragraph(project.description, styles["Subtitle"]),
                PageBreak(),
            ]

            for role, content in self.results.items():
                heading = f"{role} Analysis"
                if isinstance(content, dict) and "title" in content:
                    heading = content.get("title") or heading
                    content = {k: v for k, v in content.items() if k != "title"}
                story.append(Paragraph(heading, styles["Section"]))
                story.extend(to_flowables(content))
                story.append(PageBreak())

            # Remove last page break for cleaner output
            if story and isinstance(story[-1], PageBreak):
                story.pop()

            doc.build(story, onFirstPage=draw_border, onLaterPages=draw_border)
            logger.info("PDF exported to %s", out_path)
        except Exception as e:
            logger.exception("Failed to export PDF: %s", e)
            raise


# ─── CLI ─────────────────────────────────────────────────────────────────
app = typer.Typer()

@app.command()
def main(
    name: str = typer.Option(..., "--name", "-n", help="Project name"),
    description: str = typer.Option(..., "--description", "-d", help="Full project description"),
    project_type: str = typer.Option(..., "--type", "-t", help="Project type"),
    budget: str = typer.Option(..., "--budget", help="Budget range, e.g. $25k-$50k"),
    timeline: str = typer.Option(..., "--timeline", help="Expected timeline, e.g. 3-4 months"),
    priority: str = typer.Option(..., "--priority", help="High | Medium | Low"),
    output: Path = typer.Option(Path("proposal.pdf"), "--output", "-o", help="Output PDF path"),
    timeout: float = typer.Option(DEFAULT_TIMEOUT, "--timeout", help="Agent timeout in seconds"),
) -> None:
    """CLI entrypoint to generate a project proposal PDF."""
    try:
        project = Project(
            name=name,
            description=description,
            project_type=project_type,
            budget=budget,
            timeline=timeline,
            priority=priority,
        )
    except ValidationError as ve:
        logger.error("Project validation failed: %s", ve)
        raise typer.Exit(code=1)

    agency = B2BAgency(timeout=timeout)
    try:
        results = asyncio.run(agency.run_pipeline(project))
        agency.export_pdf(project, output)
        typer.echo(f"✅ Proposal generated: {output}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
