"""
Prompt builder for AD episode extraction & personalization using a LLM to test (e.g., Gemini).

Usage examples
--------------
1) Build a prompt from a JSON file and print it:
   $ python prompt.py build --input data.json

2) Run (build prompt -> call LLM placeholder -> print response):
   $ python prompt.py run --input data.json

3) Preview the built-in few-shot prompt only:
   $ python prompt.py preview-fewshot

Input schema (data.json)
------------------------
{
  "patient_context": {
    "id": "P045",
    "age": 78,
    "week": 3
  },
  "segment": {
    "start": "2024-03-15 14:20:00",
    "end":   "2024-03-15 14:30:00"
  },
  "activities": [
    {"action": "Take",  "ts": "2024-03-15 14:20:15"},
    {"action": "Stand", "ts": "2024-03-15 14:20:20"},
    ...
  ],
  "micro_patterns": ["Repetitive take→put cycles (4x)", "prolonged standing periods"],
  "meso_patterns": ["3 incomplete object manipulation sequences"],
  "macro_context": "2.7σ deviation in afternoon activity",
  "dimensional_scores": {
    "circadian_sigma": 1.2,
    "task_sigma": 3.8,
    "movement_sigma": 2.1,
    "social_sigma": 0.8
  },
  "medical_context": [
    {"id": 1, "cite": "Traykov et al. (2005)", "title": "Perseverative behavior in AD"},
    {"id": 2, "cite": "Dixon et al. (1999)",  "title": "Object manipulation deficits in AD"}
  ]
}

The prompt enforces a **fixed, evaluation-friendly output format**.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any

# ==========================
# Data Models
# ==========================

@dataclass
class PatientContext:
    id: str
    age: int
    week: int

@dataclass
class Segment:
    start: str  # ISO-like: YYYY-MM-DD HH:MM:SS
    end: str

@dataclass
class Activity:
    action: str
    ts: str  # ISO-like timestamp

@dataclass
class DimensionalScores:
    circadian_sigma: float
    task_sigma: float
    movement_sigma: float
    social_sigma: float

# ==========================
# Few-shot (from your example)
# ==========================

FEWSHOT_INPUT = r"""
Example of LLM Input Structure
Patient Context: ID: P045, Age: 78, Week: 3
Anomalous Segment: 2024-03-15 14:20:00 - 14:30:00 (10 minutes)
Full Activity Sequence:
Take(14:20:15)→Stand(14:20:20)→Walk(14:20:45)→Put(14:21:00)→Walk(14:21:15)→Take(14:21:30)→Stand(14:21:45)→Walk(14:22:00)→Take(14:22:30)→Put(14:22:45)→Take(14:23:00) ...
Micro patterns: Repetitive take→put cycles (4x), prolonged standing periods
Meso patterns: 3 incomplete object manipulation sequences
Macro context: 2.7σ deviation in afternoon activity
Dimensional Scores: Circadian: 1.2σ, Task: 3.8σ, Movement: 2.1σ, Social: 0.8σ
Medical Context:
[1] Traykov et al. (2005): Perseverative behavior in AD
[2] Dixon et al. (1999): Object manipulation deficits in AD
"""

FEWSHOT_OUTPUT = r"""
1. AD Manifestation Episode Extraction:
- Moment 1: 14:20:20-14:21:45 (confused object handling - take without purpose)
- Moment 2: 14:21:30-14:23:00 (perseverative take/put behavior - 4 repetitions)
- Moment 3: 14:27:00-14:28:30 (random walking - no clear destination)
Total: 4.5 mins of AD manifestation episodes within 10-minute segment

2. Clinical Interpretation:
Based on [Traykov et al., 2005], the repetitive take/put cycles indicate perseveration typical of executive dysfunction. The purposeless object manipulation aligns with apraxia patterns described in [Dixon et al., 1999].

3. Personalization Updates:
- Increase task completion weight: α_tc: 0.25 → 0.40
- Adjust other weights proportionally to maintain Σα_d = 1
"""

# ==========================
# Prompt Template
# ==========================

def _fmt_dt(dt: str) -> str:
    """Validate and return the datetime string unchanged if valid."""
    try:
        datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        raise ValueError(f"Invalid datetime format: '{dt}'. Expected 'YYYY-MM-DD HH:MM:SS'.") from e
    return dt

def _fmt_seq(activities: List[Activity]) -> str:
    parts = [f"{a.action}({_time_only(a.ts)})" for a in activities]
    return "→".join(parts)

def _time_only(ts: str) -> str:
    _fmt_dt(ts)
    return ts.split(" ")[1]

def build_prompt(
    patient: PatientContext,
    segment: Segment,
    activities: List[Activity],
    micro_patterns: List[str],
    meso_patterns: List[str],
    macro_context: str,
    scores: DimensionalScores,
    medical_context: List[Dict[str, Any]],
    fewshot: bool = True,
) -> str:
    """Build a single-text prompt for the LLM."""
    start = _fmt_dt(segment.start)
    end = _fmt_dt(segment.end)

    # Build references
    refs_lines = []
    for ref in medical_context:
        idx = ref.get("id")
        cite = ref.get("cite", "[missing cite]")
        title = ref.get("title", "[missing title]")
        refs_lines.append(f"[{idx}] {cite}: {title}")
    refs_text = "\n".join(refs_lines) if refs_lines else "(none provided)"

    seq_text = _fmt_seq(activities)
    micro_text = ", ".join(micro_patterns) if micro_patterns else "(none)"
    meso_text = ", ".join(meso_patterns) if meso_patterns else "(none)"

    header = (
        "You are a clinical research assistant LLM helping extract Alzheimer's-related behavior episodes, "
        "offer concise clinical interpretation grounded in the supplied citations, and propose personalization weight updates.\n\n"
        "Follow ALL rules:\n"
        "- Output EXACTLY the three sections shown below (1., 2., 3.) with the given headings.\n"
        "- Use local times in HH:MM:SS within the segment.\n"
        "- Reference citations only by bracketed keys provided in 'Medical Context' (e.g., [Traykov et al., 2005]).\n"
        "- Keep the clinical interpretation to <= 6 sentences total.\n"
        "- Personalization updates must keep Σα_d = 1 and state both the changed weight and proportional adjustments.\n"
        "- Do NOT invent citations beyond those provided.\n"
        "- Do NOT include any content outside the three sections.\n"
    )

    body = f"""
INPUT
-----
Patient Context: ID: {patient.id}, Age: {patient.age}, Week: {patient.week}
Anomalous Segment: {start} - {end}
Full Activity Sequence:\n{seq_text}
Micro patterns: {micro_text}
Meso patterns: {meso_text}
Macro context: {macro_context}
Dimensional Scores: Circadian: {scores.circadian_sigma}σ, Task: {scores.task_sigma}σ, Movement: {scores.movement_sigma}σ, Social: {scores.social_sigma}σ
Medical Context:\n{refs_text}

REQUIRED OUTPUT FORMAT
----------------------
1. AD Manifestation Episode Extraction:
- Moment k: HH:MM:SS-HH:MM:SS (brief label)
Total: <X> mins of AD manifestation episodes within <segment-length> segment

2. Clinical Interpretation:
<1–2 short paragraphs grounded in provided citations>

3. Personalization Updates:
- α_tc: <old> → <new>
- Adjust remaining weights proportionally to keep Σα_d = 1 (show final α list)
"""

    fewshot_block = ""
    if fewshot:
        fewshot_block = (
            "\nFEW-SHOT EXAMPLE (for style and brevity)\n" +
            "--------------------------------------\n" +
            FEWSHOT_INPUT.strip() + "\n\nEXPECTED STYLE:\n" + FEWSHOT_OUTPUT.strip() + "\n"
        )

    return header + body + fewshot_block

# ==========================
# LLM Caller (Placeholder)
# ==========================

def call_gemini(prompt: str) -> str:
    """Placeholder for Gemini call. Implement this before using `run`.

    Example (uncomment and fill in your key):

    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])  # or set directly
    model = genai.GenerativeModel("gemini-1.5-pro")
    resp = model.generate_content(prompt)
    return resp.text
    """
    raise NotImplementedError(
        "Implement Gemini call here (see docstring for a minimal snippet)."
    )

# ==========================
# CLI Utilities
# ==========================

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_activities(items: List[Dict[str, Any]]) -> List[Activity]:
    return [Activity(action=i["action"], ts=i["ts"]) for i in items]


def _parse_patient(d: Dict[str, Any]) -> PatientContext:
    return PatientContext(id=d["id"], age=int(d["age"]), week=int(d["week"]))


def _parse_segment(d: Dict[str, Any]) -> Segment:
    return Segment(start=d["start"], end=d["end"])


def _parse_scores(d: Dict[str, Any]) -> DimensionalScores:
    return DimensionalScores(
        circadian_sigma=float(d["circadian_sigma"]),
        task_sigma=float(d["task_sigma"]),
        movement_sigma=float(d["movement_sigma"]),
        social_sigma=float(d["social_sigma"]),
    )


def cmd_build(args: argparse.Namespace) -> None:
    data = _load_json(args.input)
    prompt = build_prompt(
        patient=_parse_patient(data["patient_context"]),
        segment=_parse_segment(data["segment"]),
        activities=_parse_activities(data.get("activities", [])),
        micro_patterns=data.get("micro_patterns", []),
        meso_patterns=data.get("meso_patterns", []),
        macro_context=data.get("macro_context", ""),
        scores=_parse_scores(data["dimensional_scores"]),
        medical_context=data.get("medical_context", []),
        fewshot=not args.no_fewshot,
    )
    print(prompt)


def cmd_run(args: argparse.Namespace) -> None:
    data = _load_json(args.input)
    prompt = build_prompt(
        patient=_parse_patient(data["patient_context"]),
        segment=_parse_segment(data["segment"]),
        activities=_parse_activities(data.get("activities", [])),
        micro_patterns=data.get("micro_patterns", []),
        meso_patterns=data.get("meso_patterns", []),
        macro_context=data.get("macro_context", ""),
        scores=_parse_scores(data["dimensional_scores"]),
        medical_context=data.get("medical_context", []),
        fewshot=not args.no_fewshot,
    )
    # Call LLM (you must implement call_gemini first)
    response = call_gemini(prompt)
    print(response)


def cmd_preview_fewshot(_: argparse.Namespace) -> None:
    print("FEW-SHOT INPUT:\n" + FEWSHOT_INPUT)
    print("\nFEW-SHOT OUTPUT STYLE:\n" + FEWSHOT_OUTPUT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt builder + Gemini caller (placeholder)")
    sub = parser.add_subparsers(required=True)

    p_build = sub.add_parser("build", help="Build prompt from JSON and print")
    p_build.add_argument("--input", required=True, help="Path to input JSON")
    p_build.add_argument("--no-fewshot", action="store_true", help="Disable few-shot block")
    p_build.set_defaults(func=cmd_build)

    p_run = sub.add_parser("run", help="Build prompt then call LLM (implement call_gemini first)")
    p_run.add_argument("--input", required=True, help="Path to input JSON")
    p_run.add_argument("--no-fewshot", action="store_true", help="Disable few-shot block")
    p_run.set_defaults(func=cmd_run)

    p_prev = sub.add_parser("preview-fewshot", help="Print the built-in few-shot example")
    p_prev.set_defaults(func=cmd_preview_fewshot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()