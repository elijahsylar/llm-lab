"""
coach.py — PHT coach helpers for LLM Lab.

- build_coach_prompt(protocol_context): server-side system-prompt builder. The
  PHT frontend builds its own (buildCoachPrompt.ts) and normally sends the final
  string as `system_prompt`; this is the fallback for callers that send
  structured `protocol_context` instead of a prepared prompt. The two are kept
  intentionally equivalent.

- messages_to_training_pairs(messages): convert a chat turn into the
  {"instruction","output"} JSONL rows that finetune/train.py consumes.
"""
from __future__ import annotations

import re


def _fmt_value(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, dict):
        return "(structured)"
    return str(v)


def _summarize_entry(entry: dict, metrics: list[dict]) -> str:
    name_by_id = {m.get("id"): m.get("name", m.get("id")) for m in metrics}
    values = entry.get("values", {}) or {}
    parts = []
    for mid, val in values.items():
        label = name_by_id.get(mid, mid)
        parts.append(f"{label}={_fmt_value(val)}")
    body = ", ".join(parts) if parts else "(no values)"
    return f"- {entry.get('date', '?')}: {body}"


ENTRY_EMISSION_SPEC = (
    "When — and only when — you make a concrete, loggable recommendation that the "
    "user is ready to do or has just done, emit a fenced block exactly like this:\n"
    "```entry\n"
    '{ "values": { "<metric_id>": <value>, ... }, "notes": "optional short note" }\n'
    "```\n"
    "Use the metric ids listed above as keys. Booleans are true/false, counters and "
    "sliders are numbers, text metrics are strings. Emit at most one entry block per "
    "reply, and never emit one during general discussion or when you are only asking a "
    "question. Outside that block, write normal conversational prose."
)


def _format_prescription(p: dict) -> str:
    """Render an exercise prescription dict readably (e.g. '3 × 8 @ 20 lb')."""
    parts = []
    sets = p.get("sets")
    reps = p.get("reps")
    duration = p.get("duration")
    if sets is not None and reps is not None:
        parts.append(f"{sets} × {reps}")
    elif sets is not None and duration is not None:
        parts.append(f"{sets} × {duration}s")
    elif reps is not None:
        parts.append(f"{reps} reps")
    elif duration is not None:
        parts.append(f"{duration}s")
    weight = p.get("weight")
    if isinstance(weight, (int, float)) and weight > 0:
        parts.append(f"@ {weight} lb")
    out = " ".join(parts) if parts else "—"
    if p.get("tempo"):
        out += f" · {p['tempo']}"
    if p.get("notes"):
        out += f" ({p['notes']})"
    return out


def _format_daily_targets(today_targets: dict, metrics: list[dict]) -> list[str]:
    """Render today's dailyTargets entry as readable lines (not raw JSON)."""
    name_by_id = {m.get("id"): m.get("name", m.get("id")) for m in metrics}
    out = []
    for mid, val in today_targets.items():
        label = name_by_id.get(mid, mid)
        if isinstance(val, dict):
            out.append(f"  - {label}: {_format_prescription(val)}")
        else:
            out.append(f"  - {label}: {val}")
    return out


def build_coach_prompt(ctx: dict) -> str:
    """Build the coach system prompt from a protocol context object.

    ctx keys (all optional, best-effort):
      name, description, kind, startDate, dayNumber,
      currentPhase: { label/name, week, detail, dayInPhase, totalDays, targets },
      metrics: [{id, name, type, min, max, step, unit}],
      recentEntries: [{date, values}],
      coachConfig: { constraints, personality }
    """
    name = ctx.get("name", "this protocol")
    lines: list[str] = []

    lines.append(
        f"You are a dedicated coach for the habit-tracking protocol \"{name}\". "
        "Stay focused on this protocol's domain — do not drift into unrelated topics. "
        "Be conversational and encouraging, not clinical. Keep replies short unless the "
        "user asks for detail."
    )

    desc = ctx.get("description")
    if desc:
        lines.append(f"\nProtocol overview: {desc}")

    day_number = ctx.get("dayNumber")
    if day_number is not None:
        lines.append(f"\nToday is day {day_number} of the protocol "
                     f"(start date {ctx.get('startDate', '?')}).")

    phase = ctx.get("currentPhase")
    if phase:
        plabel = phase.get("label") or phase.get("name") or "current phase"
        seg = f"\nCurrent phase: {plabel}"
        if phase.get("week") is not None:
            seg += f" (week {phase['week']})"
        if phase.get("dayInPhase") is not None and phase.get("totalDays"):
            seg += f", day {phase['dayInPhase']} of {phase['totalDays']}"
        lines.append(seg)
        if phase.get("detail"):
            lines.append(str(phase["detail"]))
        targets = phase.get("targets") or {}
        if targets:
            name_by_id = {m.get("id"): m.get("name", m.get("id"))
                          for m in ctx.get("metrics", [])}
            tparts = []
            for mid, t in targets.items():
                label = name_by_id.get(mid, mid)
                if not isinstance(t, dict):
                    tparts.append(f"{label}={t}")
                elif t.get("required") is not None:
                    tparts.append(f"{label}={t['required']}")
                elif t.get("max") is not None:
                    tparts.append(f"{label} ≤ {t['max']}")
                elif t.get("min") is not None:
                    tparts.append(f"{label} ≥ {t['min']}")
                else:
                    tparts.append(label)
            lines.append("Targets this phase: " + "; ".join(tparts))

        # Today's specific prescription (dailyTargets[today]), if the protocol
        # uses one. Skipped when absent (not all protocols are day-prescribed).
        today_targets = phase.get("todayTargets")
        if isinstance(today_targets, dict) and today_targets:
            tt_lines = _format_daily_targets(
                today_targets, ctx.get("metrics", []) or []
            )
            if tt_lines:
                lines.append(
                    "\nTODAY'S PRESCRIBED WORKOUT:\n" + "\n".join(tt_lines)
                )

    metrics = ctx.get("metrics", []) or []
    if metrics:
        mlines = []
        for m in metrics:
            bits = f"  - {m.get('id')} ({m.get('name', m.get('id'))}): {m.get('type', '?')}"
            rng = []
            if m.get("min") is not None:
                rng.append(f"min {m['min']}")
            if m.get("max") is not None:
                rng.append(f"max {m['max']}")
            if m.get("unit"):
                rng.append(str(m["unit"]))
            if rng:
                bits += " [" + ", ".join(rng) + "]"
            mlines.append(bits)
        lines.append("\nLoggable metrics (use these ids when logging):\n" + "\n".join(mlines))

    recent = ctx.get("recentEntries", []) or []
    if recent:
        rlines = [_summarize_entry(e, metrics) for e in recent[-14:]]
        lines.append("\nRecent entries (last 14 days):\n" + "\n".join(rlines))
        lines.append(
            "Use this history to track progress and avoid repeating what was just done "
            "(e.g. don't prescribe yesterday's workout again)."
        )
    else:
        lines.append("\nThere are no recent entries yet — this is a fresh start.")

    cc = ctx.get("coachConfig") or {}
    if cc.get("constraints"):
        lines.append(f"\nHard constraints you must respect: {cc['constraints']}")
    if cc.get("personality"):
        lines.append(f"\nCoaching style: {cc['personality']}")

    lines.append("\n" + ENTRY_EMISSION_SPEC)

    return "\n".join(lines)


_ENTRY_FENCE_RE = re.compile(r"```entry\b.*?```", re.DOTALL)


def messages_to_training_pairs(messages: list[dict]) -> list[dict]:
    """Pair each user message with the assistant reply that follows it.

    Returns a list of {"instruction", "output"} dicts ready to append to a
    train.py JSONL dataset. The ```entry``` blocks are intentionally KEPT in the
    output so the model learns to emit them.
    """
    pairs: list[dict] = []
    pending_user: str | None = None
    for msg in messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None and content:
            pairs.append({"instruction": pending_user, "output": content})
            pending_user = None
    return pairs
