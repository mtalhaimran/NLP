#!/usr/bin/env python3
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

import typer
from pydantic import BaseModel, Field, ValidationError
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
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
    canvas.saveState()
    canvas.setStrokeColor(colors.lightgrey)
    canvas.rect(
        0.5 * inch,
        0.5 * inch,
        doc.pagesize[0] - inch,
        doc.pagesize[1] - inch,
        stroke=1,
        fill=0,
    )
    canvas.restoreState()

# ─── PROMPT TEMPLATES ──────────────────────────────────────────────────
CEO_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        '''
You are the Chief Strategy Officer. Given the project data, perform a detailed Go/No-Go analysis. Provide output strictly as JSON matching this schema:
{{
  "decision": "<GO|NO_GO>",
  "key_risks": [ "risk1", "risk2", "risk3" ],
  "opportunities": [ "op1", "op2", "op3" ],
  "recommendations": [ {{ "title": "Recommendation title", "detail": "Recommendation detail" }} ]
}}
'''    ),
    HumanMessagePromptTemplate.from_template(
        '''
Project Data:
```json
{project}
```
Respond only with the JSON above, without additional commentary.
'''    ),
])

CTO_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        '''
You are the CTO. Based on the project data and CEO analysis, propose a robust system architecture and technology stack. Your response should include:
1. A bulleted list describing the overall architecture.
2. A JSON object with keys:
   "architecture": "<brief description>",
   "components": [ {{ "name": "<component name>", "purpose": "<component purpose>" }} ],
   "scalability_plan": "<how the system will scale>"
'''    ),
    HumanMessagePromptTemplate.from_template(
        '''
Project Data:
```json
{project}
```
CEO Analysis:
```json
{CEO}
```'''
    ),
])

PM_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        '''
You are the Product Manager. Craft a detailed three-phase roadmap using Markdown. Use H2 headers: '## MVP', '## Growth', and '## Scale'. Under each header, include bullet points for objectives and key deliverables.
'''    ),
    HumanMessagePromptTemplate.from_template(
        '''
Project Data:
```json
{project}
```
CTO Specification:
```json
{CTO}
```'''
    ),
])

DEV_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        '''
You are the Lead Developer. For each phase of the roadmap, generate a JSON section containing:
 - "tasks": a list of task descriptions,
 - "ci_cd": an object with "tool" and "pipeline_overview" fields.
Provide valid JSON only.
'''    ),
    HumanMessagePromptTemplate.from_template(
        '''
Roadmap:
```markdown
{PM}
```
'''    ),
])

MARKETING_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        '''
You are the Marketing Manager. Using the product roadmap, craft a concise go-to-market plan. Include bullet points for target audience, primary channels, and messaging themes.
'''
    ),
    HumanMessagePromptTemplate.from_template(
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

CLIENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        '''
You are the Client Success Manager. Based on the implementation plan, draft JSON containing:
 - "onboarding_process": an array of steps with "step" and "description",
 - "retention_strategy": an array of strategies,
 - "feedback_loop": an array of feedback mechanisms.
Provide JSON only.
'''    ),
    HumanMessagePromptTemplate.from_template(
        '''
Implementation Details:
```json
{DEV}
```'''
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
        try:
            doc = SimpleDocTemplate(
                str(out_path), pagesize=letter,
                leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                topMargin=1 * inch, bottomMargin=1 * inch,
            )
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle("DocTitle", parent=styles["Heading1"], fontSize=24, leading=28, spaceAfter=24))
            styles.add(ParagraphStyle("Section", parent=styles["Heading2"], fontSize=16, leading=20, spaceBefore=12, spaceAfter=6))
            styles["Normal"].fontSize = 11
            styles["Normal"].leading = 14

            story = [Paragraph(project.name, styles["DocTitle"])]
            for role, content in self.results.items():
                story.append(PageBreak())
                story.append(Paragraph(f"{role} Analysis", styles["Section"]))
                story.append(Preformatted(json.dumps(content, indent=2), styles["Normal"]))
                story.append(Spacer(1, 12))

            doc.build(story, onFirstPage=draw_border, onLaterPages=draw_border)
            logger.info(f"PDF exported to {out_path}")
        except Exception as e:
            logger.exception(f"Failed to export PDF: {e}")
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