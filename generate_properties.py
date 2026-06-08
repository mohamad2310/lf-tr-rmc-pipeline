from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from lf_analysis import analyze_lf_code
from llm_client import BaseLLMClient, PromptTooLargeError
from prompt_utils import MAX_PROMPT_CHARS, write_prompt_debug


PROPERTY_PROMPT = """You are deriving candidate verification properties from Lingua Franca code.

Your scope is only the LF program itself. Do not generate Timed Rebeca code. Do not generate AFRA property files.

Task:
1. Read the LF program carefully.
2. Derive candidate safety properties grounded directly in the LF code.
3. For each property, explicitly classify whether it is:
   - directly safe to treat as a state invariant,
   - sensitive to mapping/scheduling/output-event details,
   - too strong for plain invariant checking.
4. Be conservative. If a claim depends on timing order, output events, next-state behavior, end-to-end effects, or initialization assumptions, say so.
5. Do not invent behavior not justified by the LF code.
6. If the LF model contains recurring or periodic behavior and a state variable appears potentially unbounded, do not invent bounded invariants.
   In that case:
   - keep only sound weak invariants if necessary,
   - explicitly say when stronger bounded invariants would require a separate analysis-oriented bounded abstraction,
   - and do not hide this by returning only one trivial property without explanation.

7. If only weak or trivial invariant candidates are justified, make that clear in the Description and Confidence fields.
   Do not present a trivially small candidate set as if it were a strong verification basis.
   Briefly explain whether the weakness comes from:
   - potentially unbounded recurring behavior,
   - lack of code-enforced bounds,
   - mapping-sensitive behavior,
   - or initialization/scheduling dependence.

8. Do not output negative review notes as properties.
   For example, do not create a property whose assertion says "not suitable as a plain invariant".
   If a claim is unsuitable, use that as explanation, but still separately output any positive local invariants that are suitable.

9. If a local state variable is initialized to a nonnegative value and all code paths that can reduce it are causally paired with earlier increments or earlier enabling events, consider nonnegativity as a candidate local invariant.
   If you are unsure, mark it MEDIUM confidence, but do not discard it only because the overall model is periodic.



If the LF code does not support many strong state invariants, it is acceptable to return only a small set of weak candidates.
But in that case, explain clearly why the candidate set is small.
Return one block per property in exactly this format:
---
Property ID: P<n>
Description: <natural language>
Formal assertion: <state-based assertion or invariant candidate>
Applies to: <reactor(s), port(s), or state variable(s)>
Confidence: HIGH / MEDIUM / LOW - <reason>
Flag: <NONE | REQUIRES_MAPPING_VERIFICATION | TOO_STRONG | INITIALIZATION_ASSUMPTION>
Initialization notes: <optional note, or NONE>
---

LF structure summary:
{lf_summary}

LF program:
{lf_code}
"""

BLOCK_SPLIT_RE = re.compile(r"(?m)^\s*---\s*$")
FIELD_LABEL_RE = re.compile(r"^(Property ID|Description|Formal assertion|Applies to|Confidence|Flag|Initialization notes):\s*(.*)$")

TEMPORAL_TERMS = (
    "eventually",
    "until",
    "always eventually",
    "end-to-end",
    "response time",
    "every request",
    "next-state",
    "next state",
    "next step",
    "temporal",
    "liveness",
)
TEMPORAL_REGEXES = (
    re.compile(r"\bwithin\s+\d+(?:\.\d+)?\s*(?:ms|msec|millisecond|milliseconds|s|sec|second|seconds|us|usec|microsecond|microseconds|ns|nsec|nanosecond|nanoseconds)\b"),
    re.compile(r"\bafter\s+\d+(?:\.\d+)?\s*(?:ms|msec|millisecond|milliseconds|s|sec|second|seconds|us|usec|microsecond|microseconds|ns|nsec|nanosecond|nanoseconds)\b"),
    re.compile(r"\bbefore\s+\d+(?:\.\d+)?\s*(?:ms|msec|millisecond|milliseconds|s|sec|second|seconds|us|usec|microsecond|microseconds|ns|nsec|nanosecond|nanoseconds)\b"),
)
MAPPING_SENSITIVE_TERMS = (
    "output",
    "outputs",
    "port",
    "emits",
    "emit",
    "event",
    "message",
    "reaction order",
    "ordering",
    "scheduling",
    "delivery",
    "next-state",
    "next state",
)
EVENT_DEPENDENT_TERMS = (
    "on every execution",
    "after any",
    "after every",
    "upon receiving",
    "when an input event arrives",
    "emitted",
    "current input",
    "current payload",
    "reaction(",
)
INITIALIZATION_TERMS = (
    "initial",
    "initially",
    "startup",
    "before first",
    "uninitialized",
    "default value",
)


@dataclass
class CandidateProperty:
    property_id: str
    description: str
    formal_assertion: str
    applies_to: str
    confidence: str
    confidence_reason: str
    flag: Optional[str] = None
    classification: str = "direct-safe"
    needs_initialization_assumption: bool = False
    notes: str = ""


@dataclass
class PropertyGenerationResult:
    lf_code: str
    prompt: str
    raw_response: str
    properties: List[CandidateProperty]
    provider: str
    model: str

    def to_dict(self) -> dict:
        return {
            "lf_code": self.lf_code,
            "prompt": self.prompt,
            "raw_response": self.raw_response,
            "properties": [asdict(prop) for prop in self.properties],
            "provider": self.provider,
            "model": self.model,
        }

    def save_json(self, output_path: str | Path) -> None:
        Path(output_path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def to_prompt_summary(self) -> str:
        grouped: dict[str, list[CandidateProperty]] = {"direct-safe": [], "mapping-sensitive": [], "too-strong": []}
        for prop in self.properties:
            grouped.setdefault(prop.classification, []).append(prop)
        lines: list[str] = []
        for classification in ("direct-safe", "mapping-sensitive", "too-strong"):
            lines.append(f"{classification}:")
            if not grouped.get(classification):
                lines.append("- none")
                continue
            for prop in grouped[classification]:
                notes = prop.notes or "None"
                lines.append(
                    f"- {prop.property_id}: {prop.description} | assertion={prop.formal_assertion} | "
                    f"applies_to={prop.applies_to} | confidence={prop.confidence} ({prop.confidence_reason}) | "
                    f"flag={prop.flag or 'NONE'} | init_assumption={prop.needs_initialization_assumption} | notes={notes}"
                )
        return "\n".join(lines)


def build_prompt_sections(lf_code: str) -> dict[str, str]:
    return {
        "lf_summary": summarize_lf_for_properties(lf_code),
        "lf_code": lf_code.strip(),
    }


def build_prompt(lf_code: str) -> str:
    sections = build_prompt_sections(lf_code)
    return PROPERTY_PROMPT.format(
        lf_summary=sections["lf_summary"],
        lf_code=sections["lf_code"],
    )


def parse_properties(raw_response: str, *, lf_code: str = "") -> List[CandidateProperty]:
    uninitialized_states = extract_uninitialized_states(lf_code)
    properties: list[CandidateProperty] = []
    for block in _iter_property_blocks(raw_response):
        fields = _parse_block_fields(block)
        property_id = fields.get("Property ID", "").strip()
        if not property_id:
            continue
        confidence_value, confidence_reason = _parse_confidence(fields.get("Confidence", ""))
        classification, init_assumption, notes = classify_property(fields=fields, uninitialized_states=uninitialized_states)
        properties.append(
            CandidateProperty(
                property_id=property_id,
                description=fields.get("Description", "").strip(),
                formal_assertion=fields.get("Formal assertion", "").strip(),
                applies_to=fields.get("Applies to", "").strip(),
                confidence=confidence_value,
                confidence_reason=confidence_reason,
                flag=_normalize_flag(fields.get("Flag", "")),
                classification=classification,
                needs_initialization_assumption=init_assumption,
                notes=notes,
            )
        )
    return properties

def _looks_like_local_bookkeeping_invariant(
    assertion: str,
    applies_to: str,
    description: str,
) -> bool:
    text = " ".join([assertion or "", applies_to or "", description or ""]).lower()

    if any(
        bad_token in text
        for bad_token in (
            "eventually",
            "end-to-end",
            "response time",
            "sink.received == source.value",
        )
    ):
        return False

    if _looks_like_timed_within_phrase(text):
        return False

    indicators = (
        "count >= 0",
        "nonnegative",
        "non-negative",
        "mode",
        "binary",
        "within {",
        "local state",
        "bookkeeping",
        "written only by",
        "only updated by",
    )

    return any(token in text for token in indicators)


def classify_property(*, fields: dict[str, str], uninitialized_states: Iterable[str]) -> tuple[str, bool, str]:
    description = fields.get("Description", "")
    assertion = fields.get("Formal assertion", "")
    applies_to = fields.get("Applies to", "")
    flag_text = fields.get("Flag", "")
    initialization_notes = fields.get("Initialization notes", "")
    combined = " ".join([description, assertion, applies_to, flag_text, initialization_notes]).lower()
    notes: list[str] = []
    init_assumption = _needs_initialization_warning(combined, initialization_notes, uninitialized_states)

    local_bookkeeping = _looks_like_local_bookkeeping_invariant(
        assertion,
        applies_to,
        description,
    )

    temporal_match = _find_temporal_indicator(combined)
    if temporal_match and not local_bookkeeping:
        notes.append(f"Looks temporal or end-to-end (`{temporal_match}`).")
        if init_assumption:
            notes.append("It may also rely on initialization assumptions.")
        return "too-strong", init_assumption, " ".join(notes)

    if "requires_mapping_verification" in flag_text.lower():
        notes.append("Flagged by the model as mapping-sensitive.")
        if init_assumption:
            notes.append("It may also rely on initialization assumptions.")
        return "mapping-sensitive", init_assumption, " ".join(notes)

    if _looks_like_event_dependent_claim(description, assertion, applies_to) and not local_bookkeeping:
        notes.append("Looks event-dependent or history-dependent rather than a plain state invariant.")
        if init_assumption:
            notes.append("It may also rely on initialization assumptions.")
        return "mapping-sensitive", init_assumption, " ".join(notes)

    mapping_match = next((term for term in MAPPING_SENSITIVE_TERMS if term in combined), None)
    if mapping_match and not local_bookkeeping:
        notes.append(f"Likely mapping-sensitive because it refers to `{mapping_match}`.")
        if init_assumption:
            notes.append("It may also rely on initialization assumptions.")
        return "mapping-sensitive", init_assumption, " ".join(notes)

    if init_assumption:
        notes.append("May rely on uninitialized or startup-dependent state.")

    if "too_strong" in flag_text.lower():
        notes.append("Flagged by the model as too strong.")
        return "too-strong", init_assumption, " ".join(notes)
    
    if local_bookkeeping:
        notes.append("Looks like a local bookkeeping/state invariant.")
        return "direct-safe", init_assumption, " ".join(notes)

    notes.append("No temporal or mapping-sensitive indicators were detected.")
    return "direct-safe", init_assumption, " ".join(notes)


def extract_uninitialized_states(lf_code: str) -> set[str]:
    analysis = analyze_lf_code(lf_code or "")
    return analysis.uninitialized_state_names()


def generate_properties(
    lf_code: str,
    llm_client: BaseLLMClient,
    *,
    debug_dir: str | Path | None = None,
    debug_name: str = "candidate_properties",
) -> PropertyGenerationResult:
    prompt_sections = build_prompt_sections(lf_code)
    prompt = PROPERTY_PROMPT.format(
        lf_summary=prompt_sections["lf_summary"],
        lf_code=prompt_sections["lf_code"],
    )
    if debug_dir is not None:
        write_prompt_debug(
            debug_dir=Path(debug_dir),
            debug_name=debug_name,
            prompt=prompt,
            sections=prompt_sections,
        )
    if len(prompt) > MAX_PROMPT_CHARS:
        raise PromptTooLargeError(
            f"Property-generation prompt is too large ({len(prompt)} chars). Reduce LF input size."
        )
    raw_response = llm_client.generate(prompt)
    properties = parse_properties(raw_response, lf_code=lf_code)
    return PropertyGenerationResult(
        lf_code=lf_code,
        prompt=prompt,
        raw_response=raw_response,
        properties=properties,
        provider=llm_client.provider_name,
        model=llm_client.model_name,
    )


def _iter_property_blocks(raw_response: str) -> Iterable[str]:
    pieces = [piece.strip() for piece in BLOCK_SPLIT_RE.split(raw_response) if piece.strip()]
    for piece in pieces:
        if "Property ID:" in piece:
            yield piece


def _parse_block_fields(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    current_label: Optional[str] = None
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = FIELD_LABEL_RE.match(line)
        if match:
            current_label = match.group(1)
            fields[current_label] = match.group(2).strip()
            continue
        if current_label:
            fields[current_label] = f"{fields[current_label]} {line}".strip()
    return fields


def _parse_confidence(raw_confidence: str) -> tuple[str, str]:
    cleaned = raw_confidence.strip()
    if not cleaned:
        return "LOW", "No confidence rationale was provided."
    match = re.match(r"^(HIGH|MEDIUM|LOW)\s*[-:]\s*(.*)$", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper(), match.group(2).strip() or "No reason provided."
    upper = cleaned.upper()
    if upper in {"HIGH", "MEDIUM", "LOW"}:
        return upper, "No reason provided."
    return "LOW", cleaned


def _normalize_flag(flag_text: str) -> Optional[str]:
    cleaned = flag_text.strip()
    if not cleaned or cleaned.upper() == "NONE":
        return None
    return cleaned


def _mentions_uninitialized_state(combined_text: str, uninitialized_states: Iterable[str]) -> bool:
    for state_name in uninitialized_states:
        if re.search(rf"\b{re.escape(state_name.lower())}\b", combined_text):
            return True
    return False


def _needs_initialization_warning(combined_text: str, initialization_notes: str, uninitialized_states: Iterable[str]) -> bool:
    if any(term in combined_text for term in INITIALIZATION_TERMS):
        return True
    if initialization_notes and initialization_notes.strip().lower() not in {"", "none"}:
        return True
    return _mentions_uninitialized_state(combined_text, uninitialized_states)


def _find_temporal_indicator(text: str) -> str | None:
    for term in TEMPORAL_TERMS:
        if term in text:
            return term
    for pattern in TEMPORAL_REGEXES:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _looks_like_timed_within_phrase(text: str) -> bool:
    return TEMPORAL_REGEXES[0].search(text) is not None


def _looks_like_event_dependent_claim(description: str, assertion: str, applies_to: str) -> bool:
    text = " ".join([description or "", assertion or "", applies_to or ""]).lower()
    if any(term in text for term in EVENT_DEPENDENT_TERMS):
        return True
    if re.search(r"\b\w+'\b", text):
        return True
    return False


def summarize_lf_for_properties(lf_code: str) -> str:
    analysis = analyze_lf_code(lf_code or "")
    if not analysis.reactors:
        return "No reactor structure could be summarized from the LF source. Fall back to conservative direct reading of the program text."

    lines: list[str] = []
    for reactor in analysis.reactors:
        lines.append(f"Reactor `{reactor.name}`:")
        if reactor.timers:
            for timer in reactor.timers:
                if timer.is_periodic:
                    lines.append(
                        f"- timer `{timer.name}` has offset `{timer.offset_text or 'unspecified'}` and period `{timer.period_text}`"
                    )
                else:
                    detail = timer.offset_text or "no timing arguments"
                    lines.append(f"- timer `{timer.name}` is one-shot or non-periodic with `{detail}`")
        if reactor.actions:
            for action in reactor.actions:
                delay = action.min_delay_text or "no declared minimum delay"
                lines.append(f"- {action.kind} action `{action.name}` has `{delay}`")
        if reactor.states:
            for state in reactor.states:
                init_text = state.initializer_text if state.is_initialized else "UNINITIALIZED"
                assignment_summary = ""
                if state.assignment_expressions:
                    rendered = ", ".join(state.assignment_expressions[:4])
                    assignment_summary = f"; direct assignments seen: {rendered}"
                if state.update_expressions:
                    rendered_updates = ", ".join(state.update_expressions[:4])
                    assignment_summary += f"; updates seen: {rendered_updates}"
                lines.append(
                    f"- state `{state.name}` declared as `{state.type_text}` with initializer `{init_text}`{assignment_summary}"
                )
        if reactor.outputs:
            produced_outputs = reactor.produced_outputs()
            for output_name in reactor.outputs:
                if output_name in produced_outputs:
                    lines.append(f"- output `{output_name}` is explicitly produced by LF code via `lf_set`")
                else:
                    lines.append(f"- output `{output_name}` is declared but no explicit `lf_set({output_name}, ...)` was found")
        if reactor.reactions:
            for reaction in reactor.reactions:
                trigger_text = ", ".join(reaction.triggers) or "no triggers listed"
                effect_text = ", ".join(reaction.effects) or "no explicit effects listed"
                lines.append(f"- reaction triggered by [{trigger_text}] with declared effects [{effect_text}]")
                for action_name, delay_text in reaction.scheduled_actions:
                    lines.append(f"  schedules action `{action_name}` with delay `{delay_text}`")
    lines.append("Prefer local state invariants grounded by explicit initialization, constant assignments, finite constant domains, or direct code-enforced bounds. Avoid turning event-history claims into plain invariants.")
    return "\n".join(lines)
