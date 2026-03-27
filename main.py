"""Healthcare AI Assistant — CLI entry point."""

import os
import re
import sys
import warnings
import logging

# ── Fix Windows console encoding ────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Suppress all HuggingFace / sentence-transformers noise ──────────────────
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.config import Settings
from src.graph import build_healthcare_graph


console = Console()

# Regex to catch leaked tool-call syntax that Groq/Llama sometimes puts in content
_TOOL_NOISE = re.compile(
    r"(<function=|<\|python_tag\|>|\(function=|TOOL_CALL:)",
    re.IGNORECASE,
)


def _clean_content(content: str) -> str:
    """Strip any trailing tool-call artifacts from otherwise good content."""
    # Sometimes the model appends tool syntax after real text
    for pattern in ["<function=", "(function=", "<|python_tag|>"]:
        idx = content.find(pattern)
        if idx > 0:
            content = content[:idx]
    return content.strip()


def display_banner():
    console.print()
    console.print(Panel.fit(
        "[bold blue]Healthcare AI Assistant[/bold blue]\n"
        "[dim]Powered by LangGraph ReAct Agent + Groq[/dim]\n"
        "[dim]Type 'quit' or 'exit' to end[/dim]",
        border_style="blue",
    ))
    console.print()


def display_response(new_messages: list):
    """Display user-facing messages from a turn's output."""
    displayed = False

    for msg in new_messages:
        # Show diagnosis results from the RAG tool in a distinct block
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "analyze_symptoms":
            content = (msg.content or "").strip()
            if content:
                console.print()
                console.print(Panel(
                    Markdown(content),
                    title="[bold yellow]Diagnosis Analysis[/bold yellow]",
                    border_style="yellow",
                    expand=False,
                ))
                displayed = True
            continue

        # Show AI conversational responses
        if isinstance(msg, AIMessage):
            content = (msg.content or "").strip()
            if not content:
                continue

            # Clean any trailing tool-call artifacts
            content = _clean_content(content)
            if not content:
                continue

            # Skip pure tool-call noise
            if _TOOL_NOISE.search(content):
                continue

            console.print(f"\n[bold green]Assistant:[/bold green]")
            console.print(Markdown(content))
            console.print()
            displayed = True

    if not displayed:
        console.print("\n[bold green]Assistant:[/bold green] Let me look into that for you...\n")


def run():
    display_banner()

    # ── Load config ─────────────────────────────────────────────────────────
    try:
        settings = Settings.from_env()
    except ValueError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        console.print("[dim]Copy .env.example to .env and fill in your GROQ_API_KEY[/dim]")
        sys.exit(1)

    # ── Initialize (suppress ALL output during model download) ──────────────
    console.print("[dim]Initializing...[/dim]", end="")
    sys.stdout.flush()
    sys.stderr.flush()

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.model_name,
        temperature=0.3,
    )

    # Redirect at OS file-descriptor level to catch C-library output too
    _saved_stdout_fd = os.dup(1)
    _saved_stderr_fd = os.dup(2)
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 1)
    os.dup2(_devnull_fd, 2)
    try:
        graph = build_healthcare_graph(llm)
    finally:
        os.dup2(_saved_stdout_fd, 1)
        os.dup2(_saved_stderr_fd, 2)
        os.close(_saved_stdout_fd)
        os.close(_saved_stderr_fd)
        os.close(_devnull_fd)

    console.print(" [green]Ready![/green]\n")

    # ── Conversation loop ───────────────────────────────────────────────────
    state = {"messages": []}

    try:
        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n\n[dim]Goodbye! Take care of your health.[/dim]\n")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "bye", "q"):
                console.print("\n[dim]Goodbye! Take care of your health.[/dim]\n")
                break

            state["messages"].append(HumanMessage(content=user_input))
            prev_count = len(state["messages"])

            try:
                with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                    result = graph.invoke(state)
                    state = result

                new_messages = result["messages"][prev_count:]
                display_response(new_messages)

            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "429" in err_str:
                    console.print("\n[yellow]Rate limit reached. Please wait a moment and try again.[/yellow]\n")
                else:
                    console.print(f"\n[red]Error:[/red] {e}\n")

    except KeyboardInterrupt:
        console.print("\n\n[dim]Goodbye! Take care of your health.[/dim]\n")


def cli_main():
    run()


if __name__ == "__main__":
    run()
