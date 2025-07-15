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
from reportlab.lib.units import inch
from reportlab.lib import colors

from pdf_utils import init_styles, to_flowables

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
DEFAULT_TIMEOUT: float = 240.0
DEFAULT_RETRIES: int = 1

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
Respond concisely in JSON using this format:
{{
  "title": "<short title>",
  "sections": [
    {{"heading": "<section heading>", "content": "<60 words max>"}}
  ]
}}
Keep each section under 120 words and the full response under 400 words.
'''    ),
    human_prompt(
        '''
Project Data:
```json
{project}
```
Return only the JSON described above.
'''    ),
])

# 2. Chief Technology Officer: System architecture proposal
CTO_PROMPT = ChatPromptTemplate.from_messages([
    system_prompt(
        '''
You are the CTO. Using the project data and the CEO's analysis, propose a system architecture and technology stack.
Respond in the same JSON format as above with short sections describing the components and scalability plan.
Keep each section under 120 words and the full response under 400 words.
'''    ),
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
Return only the JSON described above.
'''
    ),
])

# 3. Product Manager: Phased roadmap
PM_PROMPT = ChatPromptTemplate.from_messages([
    system_prompt(
        '''
You are the Product Manager. Create a three-phase roadmap. Name each phase meaningfully and list key objectives.
Respond using the JSON format described above with one section per phase and an additional section for overall notes.
Keep each section under 120 words and the full response under 400 words.
'''    ),
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
Return only the JSON described above.
'''
    ),
])

# 4. Lead Developer: Task and CI/CD plan
DEV_PROMPT = ChatPromptTemplate.from_messages([
    system_prompt(
        '''
You are the Lead Developer. For each roadmap phase describe the main tasks and outline a CI/CD pipeline.
Reply in the same JSON format with one section per phase and a final section "CI/CD Pipeline".
Keep each section under 120 words and the full response under 400 words.
'''
    ),
    human_prompt(
        '''
Roadmap:
```markdown
{PM}
```
Return only the JSON described above.
'''
    ),
])

# 5. Marketing Manager: Go-to-market plan
MARKETING_PROMPT = ChatPromptTemplate.from_messages([
    system_prompt(
        '''
You are the Marketing Manager. Based on the roadmap, craft a concise go-to-market plan.
Return the plan in the same JSON format with short sections for audience, channels and messaging.
Keep each section under 120 words and the full response under 400 words.
'''    ),
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
Return only the JSON described above.
'''
    ),
])

# 6. Client Success Manager: Onboarding and retention
CLIENT_PROMPT = ChatPromptTemplate.from_messages([
    system_prompt(
        '''
You are the Client Success Manager. Using the implementation details, outline the onboarding process, retention strategy and feedback loop.
Respond using the same JSON format with concise sections for onboarding, retention and feedback.
Keep each section under 120 words and the full response under 400 words.
'''    ),
    human_prompt(
        '''
Implementation Details:
```json
{DEV}
```
Return only the JSON described above.
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
    def _trim_words(text: str, limit: int) -> str:
        """Return text limited to ``limit`` words."""
        words = text.split()
        if len(words) > limit:
            return " ".join(words[:limit]) + "..."
        return text

    @classmethod
    def _limit_json_words(cls, obj: Any, per_string: int = 60) -> Any:
        """Recursively trim all string values in ``obj``."""
        if isinstance(obj, dict):
            return {k: cls._limit_json_words(v, per_string) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._limit_json_words(v, per_string) for v in obj]
        if isinstance(obj, str):
            return cls._trim_words(obj, per_string)
        return obj

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Extract and parse the first JSON object found in text."""
        start = text.find("{")
        if start == -1:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text[start:])
        return obj

    async def run(
        self,
        project_json: str,
        context: Dict[str, Any],
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ) -> Any:
        """Invoke the LLM and return parsed JSON with optional retries."""
        base_messages = self.prompt.format_prompt(
            project=project_json,
            CEO=context.get("CEO", ""),
            CTO=context.get("CTO", ""),
            PM=context.get("PM", ""),
            DEV=context.get("DEV", ""),
        ).to_messages()

        attempt = 0
        backoff = 1.0
        while True:
            try:
                message = await asyncio.wait_for(
                    asyncio.to_thread(self.llm.invoke, base_messages),
                    timeout * backoff,
                )
                raw = getattr(message, "content", message)
                parsed = self._parse_json(raw)
                parsed = self._limit_json_words(parsed)
                logger.info(f"{self.role} responded successfully")
                return parsed
            except asyncio.TimeoutError:
                logger.error(
                    f"{self.role} agent timed out after {timeout * backoff}s"
                )
                if attempt >= retries:
                    raise
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from {self.role}")
                if attempt >= retries:
                    raise
                base_messages.append(
                    {
                        "role": "system",
                        "content": "Your last response was not valid JSON. Please answer again strictly in the required JSON format.",
                    }
                )
            attempt += 1
            backoff *= 1.5


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
        try:
            res = await agent.run(project_json, self.context, self.timeout)
        except asyncio.TimeoutError:
            logger.error(f"{role} agent timed out after {self.timeout}s")
            res = {"error": f"{role} agent timed out"}
        except Exception as e:
            logger.exception(f"{role} agent failed: {e}")
            res = {"error": str(e)}
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

        try:
            doc = SimpleDocTemplate(
                str(out_path),
                pagesize=letter,
                leftMargin=0.75 * inch,
                rightMargin=0.75 * inch,
                topMargin=1 * inch,
                bottomMargin=1 * inch,
            )

            styles = init_styles()

            story: List[Any] = [
                Paragraph(project.name, styles["DocTitle"]),
                Paragraph(project.description, styles["Subtitle"]),
                PageBreak(),
            ]

            for agent in self.agents:
                role = agent.role
                content = self.results.get(role)
                if content is None:
                    continue

                summary = json.dumps(content, indent=2, ensure_ascii=False)
                story.append(Paragraph(f"{role} Summary", styles["Section"]))
                story.append(
                    Preformatted(summary, styles["NormalLeft"], maxLineLength=80)
                )
                story.append(PageBreak())

                heading = f"{role} Details"
                if isinstance(content, dict) and "title" in content:
                    heading = content.get("title") or heading
                    body = {k: v for k, v in content.items() if k != "title"}
                else:
                    body = content
                story.append(Paragraph(heading, styles["Section"]))
                story.extend(to_flowables(body, styles))
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
