"""
The clinician-facing report writer (OpenAI, used honestly)
==========================================================
The LLM does NOT decide anything. It receives the already-computed structured
verdicts and turns them into a readable summary, grounded strictly in the
evidence passed to it. The mandatory safety line is appended in CODE, never
left to the model to remember.

This is the sponsor-friendly, non-gimmicky use of the API: turning structured
evidence into language, plus normalising messy antibiotic names.
"""
from __future__ import annotations
import json
import os

MANDATORY_DISCLAIMER = (
    "⚠️ This is a research prototype. Every result MUST be confirmed by standard "
    "laboratory antibiotic-susceptibility testing before any treatment decision. "
    "This tool is decision support only and does not make treatment decisions."
)

_SYSTEM = (
    "You are a careful clinical-microbiology report writer. You will receive a "
    "JSON list of per-antibiotic verdicts, each with a label (likely to fail / "
    "likely to work / no-call), a calibrated confidence, an evidence tier, and "
    "the supporting genes. Write a concise, neutral summary a clinician can read "
    "in 15 seconds. RULES: (1) State only what the evidence supports — never "
    "invent a mechanism or gene. (2) Clearly distinguish a KNOWN determinant "
    "from a merely statistical association. (3) Never recommend a specific "
    "treatment or dose. (4) Present no-calls as a deliberate strength, not a "
    "failure. Do not add a disclaimer; one is appended automatically."
)


def render_report(verdicts: list[dict], species: str, use_llm: bool = True) -> str:
    """verdicts: list of Verdict.to_dict(). Returns final report text."""
    if use_llm and os.getenv("OPENAI_API_KEY"):
        try:
            return _llm_report(verdicts, species)
        except Exception as e:  # fall back to deterministic template
            body = _template_report(verdicts, species) + f"\n\n(LLM unavailable: {e})"
            return body + "\n\n" + MANDATORY_DISCLAIMER
    return _template_report(verdicts, species) + "\n\n" + MANDATORY_DISCLAIMER


def _llm_report(verdicts, species):
    from openai import OpenAI

    client = OpenAI()
    msg = (f"Species: {species}\nVerdicts:\n"
           f"{json.dumps(verdicts, indent=2)}")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": msg}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    return text + "\n\n" + MANDATORY_DISCLAIMER


def _template_report(verdicts, species):
    lines = [f"Antibiotic-response summary — {species}", ""]
    for v in verdicts:
        ev = {
            "known_determinant": "known resistance determinant detected",
            "statistical_only": "statistical association only (no curated determinant)",
            "no_signal": "no resistance signal found",
        }.get(v["evidence_tier"], v["evidence_tier"])
        genes = ", ".join(v["supporting_features"]) or "none"
        lines.append(
            f"• {v['antibiotic']}: {v['label'].upper()} "
            f"(confidence {v['confidence']:.2f}; evidence: {ev}; genes: {genes})"
        )
        lines.append(f"    {v['reason']}")
    return "\n".join(lines)


def normalise_antibiotic(name: str) -> str:
    """Optional OpenAI-assisted mapping of messy drug names to canonical form.
    Falls back to a lowercase/trim if no API key is set.
    """
    canon = name.strip().lower()
    if not os.getenv("OPENAI_API_KEY"):
        return canon
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                       f"Return ONLY the canonical lowercase generic drug name for: "
                       f"'{name}'. No punctuation, no explanation."}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip().lower()
    except Exception:
        return canon
