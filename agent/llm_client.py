# agent/llm_client.py
# ─────────────────────────────────────────────────────────────
# Claude API wrapper implementing the tool-calling loop.
#
# KEY DESIGN PATTERNS (Thesis Chapter 4.4):
#   1. Bounded iteration   — MAX_AGENT_ITERATIONS prevents runaway loops
#   2. Full audit trail    — every tool call logged to JSONL
#   3. Structured errors   — every tool returns {status, message}
#   4. Parser isolation    — parse_event_from_text() is independent
#                            so parsing failures never block scoring
#   5. Dual mode           — LLM agent OR pure pipeline, same output
# ─────────────────────────────────────────────────────────────

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import anthropic
from dotenv import load_dotenv

from config import LLM_MODEL, LLM_MAX_TOKENS, MAX_AGENT_ITERATIONS, LOG_AGENT_CALLS, LOG_DIR
from agent.tool_definitions import TOOL_DEFINITIONS
from agent.tools import (
    tool_load_contacts,
    tool_run_scoring_model,
    tool_generate_top_list,
    tool_explain_scores,
)

load_dotenv()

# ── Map tool name strings → Python functions ──────────────────
TOOL_MAP: dict[str, Callable] = {
    "tool_load_contacts":      tool_load_contacts,
    "tool_run_scoring_model":  tool_run_scoring_model,
    "tool_generate_top_list":  tool_generate_top_list,
    "tool_explain_scores":     tool_explain_scores,
}

# ── System prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise supply chain analytics agent specialising in
contact database scoring for continuing education providers.

Your role is to help marketing and logistics teams identify the highest-probability
attendees for new events, using a trained predictive ML model and SCM-grounded
feature engineering.

## Mandatory workflow for every scoring request:
1. Call tool_load_contacts to verify the data loaded correctly.
2. Call tool_run_scoring_model with structured event metadata.
3. Call tool_generate_top_list to save the output file.
4. If the user requests explanations: call tool_explain_scores.
5. Summarise results concisely: contacts scored, top score, ABC distribution,
   output file path.

## Field extraction rules:
- You must extract ALL five required event fields before calling tool_run_scoring_model:
  topic, format, price, location, audience.
- If ANY required field is missing, state clearly which field is missing and ask.
  Do NOT proceed with scoring using assumed or fabricated values.
- price: extract numeric EUR value. If the user says "free", use 0.
- audience: extract as a list of job titles or roles mentioned.
- format: normalise to one of: Seminar, Congress, E-Learning, Workshop, Other.

## Error handling:
- If any tool returns status="error", report the error clearly and stop.
  Do not attempt to continue the workflow after a tool error.

## Tone: Professional, concise, results-oriented. No filler phrases.
"""


def run_agent_loop(
    user_message: str,
    contacts_path: str,
    model_path: str = "model/model.pkl",
    stream_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Execute the full agent tool-calling loop for one scoring request.

    Args:
        user_message : Event description (free text or structured).
        contacts_path: Path to the contacts CSV/Excel file.
        model_path   : Path to trained model pickle.
        stream_callback: Optional callable(str) for real-time UI progress.

    Returns:
        dict with keys:
            output_path   (str|None) : Path to the scored output file.
            explanations  (list)     : SHAP explanations if requested.
            final_response(str)      : Claude's plain-text summary.
            tool_calls    (list)     : Audit log of every tool invocation.
            status        (str)      : "ok" | "error" | "incomplete".
            error         (str|None) : Error message when status != "ok".
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Inject file paths so Claude knows them without asking
    augmented = (
        f"{user_message}\n\n"
        f"[System context]\n"
        f"contacts_path : {contacts_path}\n"
        f"model_path    : {model_path}"
    )

    messages = [{"role": "user", "content": augmented}]
    result = {
        "output_path":    None,
        "explanations":   [],
        "final_response": "",
        "tool_calls":     [],
        "status":         "ok",
        "error":          None,
    }

    def _emit(msg: str):
        if stream_callback:
            stream_callback(msg)

    def _log_tool(name: str, inp: dict, out: dict, ms: int):
        entry = {
            "run_id":    run_id,
            "timestamp": datetime.now().isoformat(),
            "tool":      name,
            "input":     inp,
            "output":    {"status": out.get("status"), "message": out.get("message", "")},
            "duration_ms": ms,
        }
        result["tool_calls"].append(entry)
        if LOG_AGENT_CALLS:
            Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
            with open(f"{LOG_DIR}agent_log.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")

    _emit(f"Agent started — run_id: {run_id}")

    for iteration in range(MAX_AGENT_ITERATIONS):

        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # ── Agent is done ─────────────────────────────────────
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    result["final_response"] = block.text
                    _emit(f"✓ Complete after {iteration + 1} iteration(s).")
            break

        # ── Agent wants to call a tool ────────────────────────
        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                name  = block.name
                inp   = block.input
                _emit(f"  → {name}({', '.join(f'{k}={repr(v)[:40]}' for k, v in inp.items())})")

                if name not in TOOL_MAP:
                    out = {"status": "error",
                           "message": f"Unknown tool '{name}'. Available: {list(TOOL_MAP)}"}
                else:
                    t0 = time.monotonic()
                    try:
                        out = TOOL_MAP[name](**inp)
                    except TypeError as e:
                        out = {"status": "error", "message": f"Bad arguments: {e}"}
                    except Exception as e:
                        out = {"status": "error", "message": str(e)}
                    _log_tool(name, inp, out, int((time.monotonic() - t0) * 1000))

                _emit(f"    ✓ {out.get('message', 'done')}")

                # Capture key outputs
                if name == "tool_generate_top_list" and out.get("status") == "ok":
                    result["output_path"] = out.get("output_path")
                if name == "tool_explain_scores" and out.get("status") == "ok":
                    result["explanations"] = out.get("explanations", [])

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     json.dumps(out)[:12_000],
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            result["status"] = "error"
            result["error"]  = f"Unexpected stop_reason: {response.stop_reason}"
            _emit(f"✗ {result['error']}")
            break

    else:
        result["status"] = "incomplete"
        result["error"]  = f"Agent exceeded {MAX_AGENT_ITERATIONS} iterations without completing."
        _emit(result["error"])

    return result


def parse_event_from_text(text: str) -> dict | None:
    """
    One-shot LLM call to extract structured event metadata from free text.

    Isolated from the main agent loop: if this fails, the UI falls back
    to the structured form without aborting the workflow.

    Args:
        text: Natural-language event description.

    Returns:
        Structured dict or None on failure.

    Example:
        "Two-day HR seminar in Frankfurt, 890 EUR, Sept 2025, for HR managers"
        → {"topic": "HR Management", "format": "Seminar", "price": 890,
           "location": "Frankfurt", "audience": ["HR Manager"],
           "date": "2025-09-01", "duration_days": 2, "online": false}
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system = (
        "Extract event metadata from the description. "
        "Respond ONLY with valid JSON — no markdown, no preamble, no trailing text. "
        "Required keys and types:\n"
        "  topic         (string)  : main subject\n"
        "  format        (string)  : Seminar | Congress | E-Learning | Workshop | Other\n"
        "  price         (number)  : EUR, 0 if not mentioned\n"
        "  location      (string)  : city/region, empty string if online or unknown\n"
        "  audience      (array)   : list of target job titles/roles, [] if unspecified\n"
        "  date          (string)  : YYYY-MM-DD, empty string if not mentioned\n"
        "  duration_days (integer) : 1 if not mentioned\n"
        "  online        (boolean) : true only if explicitly described as online/virtual"
    )

    try:
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": text}],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if model adds them despite instructions
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return None
