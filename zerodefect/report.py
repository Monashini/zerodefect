"""Shift-report agent: Muse Spark queries the inspection log through tools, looks for
patterns (time of day, location on the part, defect type) and writes an action-oriented
report for the factory owner. Falls back to a deterministic report offline."""
from __future__ import annotations

import json
import logging
import time

from .config import settings
from .meta_api import async_client
from .store import Store

log = logging.getLogger("zerodefect.report")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": "Totals for the period: parts inspected, pass, fail, review, defect rate.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_defects_by_hour",
            "description": "Parts inspected, failures and defect rate for each hour of the period.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_defect_breakdown",
            "description": "Count of each defect type, the most common locations on the part, average severity.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_failures",
            "description": "The most recent failed parts with the inspector's explanation and likely cause.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "1-30"}},
                "required": ["limit"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM = (
    "You are ZeroDefect's shift analyst for a small factory in India. Use the tools to study the "
    "inspection log, then write a short report for the owner (who is not an engineer). Look for "
    "patterns: a defect rate that rises at a certain hour, one defect type dominating, defects "
    "clustering at one location on the part (that points to a specific tool, die or fixture). "
    "Format in Markdown with these sections: **Shift summary** (2-3 lines with numbers), "
    "**What we found** (up to 4 bullets, each backed by a number), **Do this next** (up to 3 "
    "concrete actions, e.g. 'check the die at station 2 after 3 PM'). Under 220 words. Never "
    "invent data you did not get from a tool."
)


def _run_tool(store: Store, name: str, args: dict, since: float | None, product: str | None):
    if name == "get_summary":
        return store.summary(since, product=product)
    if name == "get_defects_by_hour":
        return store.by_hour(since, product=product)
    if name == "get_defect_breakdown":
        return store.defect_breakdown(since, product=product)
    if name == "get_recent_failures":
        return store.recent_failures(int(max(1, min(30, args.get("limit", 10)))), product=product)
    return {"error": f"unknown tool {name}"}


async def agent_report(store: Store, since: float | None = None, product: str | None = None, max_steps: int = 6) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Write the report for product '{product or 'all products'}'. Current time: {time.strftime('%Y-%m-%d %H:%M')}."},
    ]
    trace = []
    for _ in range(max_steps):
        resp = await async_client().chat.completions.create(
            model=settings.spark_model, messages=messages, tools=TOOLS, reasoning_effort="medium"
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"markdown": msg.content or "", "source": settings.spark_model, "trace": trace}
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            }
        )
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _run_tool(store, tc.function.name, args, since, product)
            trace.append({"tool": tc.function.name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)})
    messages.append({"role": "user", "content": "Write the final report now."})
    resp = await async_client().chat.completions.create(model=settings.spark_model, messages=messages)
    return {"markdown": resp.choices[0].message.content or "", "source": settings.spark_model, "trace": trace}


def _name(t: str) -> str:
    return "defect (type not classified offline)" if t in ("other", "", None) else t.replace("_", " ")


def local_report(store: Store, since: float | None = None, product: str | None = None) -> dict:
    s = store.summary(since, product=product)
    if not s["total_parts"]:
        return {"markdown": "No parts inspected yet.", "source": "local", "trace": []}
    hours = store.by_hour(since, product=product)
    br = store.defect_breakdown(since, product=product)
    lines = [
        "**Shift summary**",
        f"{s['total_parts']} parts inspected: {s['pass']} passed, {s['fail']} failed, {s['review']} sent for review "
        f"(defect rate {s['defect_rate_pct']}%).",
        "",
        "**What we found**",
    ]
    if br:
        top = br[0]
        loc = f", mostly at the {top['top_locations'][0]}" if top["top_locations"] and top["top_locations"][0] != "?" else ""
        lines.append(f"- Most common defect: {_name(top['defect_type'])} ({top['count']} parts{loc}).")
        if len(br) > 1:
            lines.append("- Others: " + ", ".join(f"{_name(b['defect_type'])} ({b['count']})" for b in br[1:4]) + ".")
    worst = max(hours, key=lambda h: (h["defect_rate_pct"], h["total"])) if hours else None
    if worst and len(hours) > 1 and worst["fail"]:
        lines.append(f"- Worst hour: {worst['hour'][-5:]} with a {worst['defect_rate_pct']}% defect rate.")
    lines += ["", "**Do this next**"]
    if br:
        where = br[0]["top_locations"][0] if br[0]["top_locations"] else "the defect area"
        lines.append(f"- Check the tool or fixture that touches the {where} of the part.")
    if worst and len(hours) > 1 and worst["fail"]:
        lines.append(f"- Check machine condition (temperature, wear) around {worst['hour'][-5:]}.")
    lines.append("- Keep failed parts aside for re-check; mark false alarms in ZeroDefect so it learns.")
    return {"markdown": "\n".join(lines), "source": "local", "trace": []}


async def build_report(store: Store, since: float | None = None, product: str | None = None) -> dict:
    if settings.meta_enabled and store.summary(since, product=product)["total_parts"]:
        try:
            return await agent_report(store, since, product)
        except Exception as exc:
            log.warning("Muse Spark report failed, using local report: %s", exc)
    return local_report(store, since, product)
