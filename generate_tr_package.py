from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from lf_analysis import analyze_lf_code
from llm_client import BaseLLMClient, PromptTooLargeError
from prompt_utils import MAX_PROMPT_CHARS, truncate_text, write_prompt_debug


TR_PROMPT = """You are generating a baseline Timed Rebeca verification package from Lingua Franca input plus already-derived candidate properties.

Follow these rules strictly:
1. Build the strict LF-faithful baseline first.
2. Do not add artificial keep-alive behavior unless it is explicitly requested or clearly justified and labeled.
3. Keep the baseline property file conservative. Prefer only robust assertions first.
4. If the LF model is one-shot, explain terminal deadlock rather than forcing infinite behavior.
5. If the LF model is periodic, analyze boundedness and branch reachability before choosing stronger properties.
6. Preserve LF timing semantics. If the LF model has a periodic timer or recurring logical trigger, the corresponding Rebeca rescheduling must use a positive delay that matches the LF recurrence, not `after(0)`.
7. `after(0)` is allowed only for immediate propagation between reactions or actors at the same logical time. It must not create a recurring self-loop that keeps re-triggering the same actor forever at logical time 0.
8. Model LF input-port presence explicitly. When an input arrives, store the value, set a presence flag, run the relevant reaction logic, and clear the presence flag after consumption. Do not treat stale input values as permanently present.
9. Produce outputs only when the LF code actually emits or sets them. Do not invent output behavior for ports that the LF code does not produce.
10. Every generated Rebeca `statevars` entry, including helper variables such as presence flags and pending flags, must be explicitly initialized in the constructor of that reactiveclass.
11. If an input-handling msgsrv stores input and then schedules an internal zero-delay self-reaction, guard that scheduling with a generic pending flag so repeated inputs do not enqueue duplicate internal reactions. Use a pattern like:
    - boolean react_pending;
    - in the input receiver: if (!react_pending) {{ react_pending = true; self.react() after(0); }}
    - in the internal reaction: react_pending = false;

More specifically, when analyzing periodic or recurring LF behavior, explicitly check whether any state variable is updated under:
- recurring timers,
- recurring logical actions,
- self-scheduling behavior,
- periodic environment inputs,
- or any other ongoing trigger.

For every such variable, determine whether it is:
- explicitly bounded by code,
- reset by code,
- conditionally stabilized by reachable corrective logic,
- or potentially unbounded.

If a variable is monotonically increasing, decreasing, or accumulating under recurring behavior and no explicit bound, reset, or reachable stabilizing branch is enforced by the code, treat the strict LF-faithful baseline as potentially unbounded.
12. Do not silently fix LF bugs. If you provide an analysis-oriented variant, label it clearly as not being the strict baseline.
13. Use the candidate-property classifications. `direct-safe` properties are preferred. `mapping-sensitive` and `too-strong` items must not be blindly emitted as baseline RMC/Rebeca invariants.

The candidate classification is advisory, not absolute.
If a candidate was classified as mapping-sensitive or too-strong for a broad textual reason, but the LF code and the generated TR model show that it is actually a local state invariant, you may include it in the baseline property file.
When doing this, explain in PROPERTY REVIEW why it is safe despite the original classification.

Do not confuse desired value ranges with code-enforced value ranges.

A bounded property such as:
    0 <= x <= K
may be included in the baseline only if the LF code or the generated baseline model itself enforces that upper bound through:
- an explicit guard,
- a saturating update,
- a reset,
- a reachable corrective branch,
- or another direct code-grounded mechanism.

If the upper bound is not enforced by the code, do not include it as a baseline invariant.
Instead:
- keep only the weaker sound property if one exists, or
- state clearly that stronger bounded invariants would require a separate analysis-oriented abstraction.
However, do not let this rule eliminate all useful verification structure.

If global bounded ranges are not justified, still build the baseline property file from sound local invariants whenever possible, such as:
- nonnegativity of local bookkeeping counters,
- local domain constraints such as binary modes,
- local state variables initialized to nonnegative values and only assigned from nonnegative values,
- and local invariants that do not require end-to-end delivery assumptions.

If the candidate-property classifier marked a local bookkeeping invariant as mapping-sensitive or too-strong only because its explanation mentioned scheduling, output, or reaction order, re-evaluate it from the LF code and generated TR model before discarding it.

When deciding whether a recurring state variable is effectively bounded, do not inspect only the update statement itself.
Also check whether any corrective or stabilizing branch is actually reachable in the generated model.

A controller or reactor should be treated as effectively bounding a variable only if the relevant corrective branch is reachable.
If a duplicated guard, scheduling artifact, or mapping choice makes the corrective branch unreachable, then the variable must be treated as potentially unbounded.

If the strict LF-faithful baseline is periodic and potentially unbounded, say so explicitly in MODEL ANALYSIS.

Do not hide this situation by returning only one trivial property without explanation.
Instead, explain:
- why stronger invariants are not semantically justified,
- why the reachable state space may grow,
- and whether a separate bounded analysis-oriented model may be useful if the verification goal is finite-state invariant checking rather than pure LF-faithful baseline preservation.

If the strict LF-faithful baseline would leave the verification package with only trivial invariants, do not stop at merely reporting that fact.
Also evaluate whether a separate, clearly labeled, analysis-oriented bounded abstraction is justified for finite-state invariant checking.

If you provide such an abstraction:
- keep it semantically close to the LF model,
- explain exactly what bound, saturation, or abstraction was introduced,
- keep it separate from the strict baseline,
- and do not present it as the LF-faithful baseline.
- If you include optional bounded-variant code, place it only inside section D using fenced blocks tagged `rebeca_variant` and `property_variant`.

Return exactly these sections:
A. MODEL ANALYSIS
B. BASELINE FILES
C. EXPECTED VERIFICATION OUTCOME
D. OPTIONAL SECOND ANALYSIS VERSION
E. PROPERTY REVIEW

At the end of the response, include exactly:
- one fenced code block tagged `rebeca` containing the baseline `.rebeca`
- one fenced code block tagged `property` containing the baseline `.property`
Do not include extra fenced `rebeca` or `property` blocks for optional variants.
Optional analysis-oriented bounded variants, if any, must use `rebeca_variant` and `property_variant` tags and must be clearly labeled as non-baseline artifacts.

The `.property` code block must use the full wrapped Rebeca property syntax accepted by the RMC/Rebeca property parser. Use this structure exactly:
property {{
    define {{
        name1 = (<boolean expression>);
    }}

    Assertion {{
        P1: name1;
    }}
}}
Never output only `invariant(...)` lines without the wrapper.
If an example `.property` file is provided, match its structural style exactly.

Inputs
--- LF CODE ---
{lf_code}

--- LF SEMANTICS HINTS ---
{lf_semantics}

--- STRUCTURED CANDIDATE PROPERTIES ---
{candidate_properties}

--- EXAMPLE REBECA ---
{example_rebeca}

--- EXAMPLE PROPERTY ---
{example_property}

--- PREVIOUS FAILURE CONTEXT ---
{failure_context}
"""

SECTION_RE = {
    "model_analysis": re.compile(r"A\. MODEL ANALYSIS\s*(.*?)(?=\nB\. |\Z)", re.DOTALL | re.IGNORECASE),
    "expected_outcome": re.compile(r"C\. EXPECTED VERIFICATION OUTCOME\s*(.*?)(?=\nD\. |\nE\. |\Z)", re.DOTALL | re.IGNORECASE),
    "analysis_variant": re.compile(r"D\. OPTIONAL SECOND ANALYSIS VERSION\s*(.*?)(?=\nE\. |\Z)", re.DOTALL | re.IGNORECASE),
}
REBECA_BLOCK_RE = re.compile(r"```rebeca\s*(.*?)```", re.DOTALL | re.IGNORECASE)
PROPERTY_BLOCK_RE = re.compile(r"```property\s*(.*?)```", re.DOTALL | re.IGNORECASE)
VARIANT_REBECA_BLOCK_RE = re.compile(r"```(?:rebeca_variant|rebeca-variant)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
VARIANT_PROPERTY_BLOCK_RE = re.compile(r"```(?:property_variant|property-variant)\s*(.*?)```", re.DOTALL | re.IGNORECASE)
COMPACT_CANDIDATE_PROPERTIES_CHARS = 12_000
COMPACT_EXAMPLE_REBECA_CHARS = 10_000
COMPACT_EXAMPLE_PROPERTY_CHARS = 4_000
COMPACT_FAILURE_CONTEXT_CHARS = 16_000


@dataclass
class TRPackageResult:
    prompt: str
    raw_response: str
    model_analysis: str
    rebeca_code: str
    property_code: str
    expected_outcome: str
    optional_analysis_model: Optional[str] = None
    optional_analysis_rebeca_code: Optional[str] = None
    optional_analysis_property_code: Optional[str] = None
    provider: str = ""
    model: str = ""
    property_was_normalized: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, project_dir: str | Path, project_name: str) -> Dict[str, str]:
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        rebeca_path = project_dir / f"{project_name}.rebeca"
        property_path = project_dir / f"{project_name}.property"
        json_path = project_dir / f"{project_name}.tr_package.json"
        raw_output_path = project_dir / f"{project_name}.tr_package.raw.txt"
        rebeca_path.write_text(self.rebeca_code, encoding="utf-8")
        property_path.write_text(self.property_code, encoding="utf-8")
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        raw_output_path.write_text(self.raw_response, encoding="utf-8")
        paths = {
            "rebeca": str(rebeca_path),
            "property": str(property_path),
            "json": str(json_path),
            "raw_output": str(raw_output_path),
        }
        if self.optional_analysis_rebeca_code and self.optional_analysis_property_code:
            variant_dir = project_dir / "analysis_bounded"
            variant_dir.mkdir(parents=True, exist_ok=True)
            variant_rebeca_path = variant_dir / f"{project_name}.rebeca"
            variant_property_path = variant_dir / f"{project_name}.property"
            variant_note_path = variant_dir / "WARNING.txt"
            variant_rebeca_path.write_text(self.optional_analysis_rebeca_code, encoding="utf-8")
            variant_property_path.write_text(self.optional_analysis_property_code, encoding="utf-8")
            variant_note_path.write_text(
                "This is an analysis-oriented bounded abstraction, not the strict LF-faithful baseline.",
                encoding="utf-8",
            )
            paths["analysis_variant_dir"] = str(variant_dir)
        return paths


def build_prompt_sections(
    lf_code: str,
    candidate_properties: str | dict[str, Any],
    example_rebeca: str = "",
    example_property: str = "",
    failure_context: str = "None",
    *,
    compact: bool = False,
) -> dict[str, str]:
    candidate_properties_text = _candidate_properties_text(candidate_properties, compact=compact)
    sections = {
        "lf_code": lf_code.strip() or "None",
        "lf_semantics": summarize_lf_semantics(lf_code),
        "candidate_properties": candidate_properties_text or "None",
        "example_rebeca": example_rebeca.strip() or "None",
        "example_property": example_property.strip() or "None",
        "failure_context": failure_context.strip() or "None",
    }
    if compact:
        sections["candidate_properties"] = truncate_text(sections["candidate_properties"], COMPACT_CANDIDATE_PROPERTIES_CHARS)
        sections["example_rebeca"] = truncate_text(sections["example_rebeca"], COMPACT_EXAMPLE_REBECA_CHARS)
        sections["example_property"] = truncate_text(sections["example_property"], COMPACT_EXAMPLE_PROPERTY_CHARS)
        sections["failure_context"] = truncate_text(sections["failure_context"], COMPACT_FAILURE_CONTEXT_CHARS)
    return sections


def build_prompt_from_sections(sections: dict[str, str]) -> str:
    return TR_PROMPT.format(
        lf_code=sections["lf_code"],
        lf_semantics=sections["lf_semantics"],
        candidate_properties=sections["candidate_properties"],
        example_rebeca=sections["example_rebeca"],
        example_property=sections["example_property"],
        failure_context=sections["failure_context"],
    )


def build_prompt(lf_code: str, candidate_properties: str | dict[str, Any], example_rebeca: str = "", example_property: str = "", failure_context: str = "None") -> str:
    return build_prompt_from_sections(
        build_prompt_sections(
            lf_code=lf_code,
            candidate_properties=candidate_properties,
            example_rebeca=example_rebeca,
            example_property=example_property,
            failure_context=failure_context,
        )
    )


def extract_code_sections(raw_response: str) -> Dict[str, str]:
    rebeca_blocks = [block.strip() for block in REBECA_BLOCK_RE.findall(raw_response)]
    property_blocks = [block.strip() for block in PROPERTY_BLOCK_RE.findall(raw_response)]
    variant_rebeca_blocks = [block.strip() for block in VARIANT_REBECA_BLOCK_RE.findall(raw_response)]
    variant_property_blocks = [block.strip() for block in VARIANT_PROPERTY_BLOCK_RE.findall(raw_response)]
    if not rebeca_blocks:
        raise ValueError("Expected at least one baseline ```rebeca``` block.")
    if not property_blocks:
        raise ValueError("Expected at least one baseline ```property``` block.")
    return {
        "model_analysis": _extract(SECTION_RE["model_analysis"], raw_response),
        "rebeca_code": rebeca_blocks[-1],
        "property_code": property_blocks[-1],
        "expected_outcome": _extract(SECTION_RE["expected_outcome"], raw_response),
        "analysis_variant": _extract(SECTION_RE["analysis_variant"], raw_response),
        "variant_rebeca_code": variant_rebeca_blocks[-1] if variant_rebeca_blocks else "",
        "variant_property_code": variant_property_blocks[-1] if variant_property_blocks else "",
    }


def validate_extraction(parts: Dict[str, str], *, lf_code: str = "", allow_property_normalization: bool = False) -> bool:
    rebeca_code = parts["rebeca_code"].strip()
    property_code = parts["property_code"].strip()
    property_was_normalized = False

    if allow_property_normalization:
        normalized = normalize_property_code(property_code)
        if normalized != property_code:
            property_code = normalized
            parts["property_code"] = property_code
            property_was_normalized = True

    if not rebeca_code:
        raise ValueError("LLM response did not contain a usable baseline rebeca block.")
    if not property_code:
        raise ValueError("LLM response did not contain a usable baseline property block.")

    lowered_rebeca = rebeca_code.lower()
    lowered_property = property_code.lower()
    if "reactiveclass" not in lowered_rebeca or "main" not in lowered_rebeca:
        raise ValueError("Extracted rebeca code must contain both `reactiveclass` and `main`.")
    _validate_rebeca_structure(rebeca_code)
    _validate_statevars_initialized(rebeca_code)
    _validate_no_zero_delay_self_loop_cycles(rebeca_code, lf_code=lf_code)
    _validate_presence_mapping_expectations(rebeca_code, lf_code=lf_code)
    _validate_pending_guards_for_zero_delay_internal_reactions(rebeca_code)
    missing_markers = [marker for marker in ("property", "define", "assertion") if marker not in lowered_property]
    if missing_markers:
        snippet = property_code[:500].replace("\r", "")
        raise ValueError(
            "Extracted property code must contain `property`, `define`, and `Assertion` syntax markers. "
            f"Missing: {', '.join(missing_markers)}. Extracted property block starts with:\n{snippet}"
        )
    return property_was_normalized


def generate_tr_package(
    lf_code: str,
    candidate_properties: str | dict[str, Any],
    llm_client: BaseLLMClient,
    example_rebeca: str = "",
    example_property: str = "",
    failure_context: str = "None",
    debug_dir: str | Path | None = None,
    debug_name: str = "tr_package",
    allow_property_normalization: bool = False,
) -> TRPackageResult:
    prompt_sections = build_prompt_sections(
        lf_code=lf_code,
        candidate_properties=candidate_properties,
        example_rebeca=example_rebeca,
        example_property=example_property,
        failure_context=failure_context,
    )
    prompt = build_prompt_from_sections(prompt_sections)
    if len(prompt) > MAX_PROMPT_CHARS:
        if debug_dir is not None:
            write_prompt_debug(
                debug_dir=Path(debug_dir),
                debug_name=f"{debug_name}.initial",
                prompt=prompt,
                sections=prompt_sections,
            )
        prompt_sections = build_prompt_sections(
            lf_code=lf_code,
            candidate_properties=candidate_properties,
            example_rebeca=example_rebeca,
            example_property=example_property,
            failure_context=failure_context,
            compact=True,
        )
        prompt = build_prompt_from_sections(prompt_sections)
    if debug_dir is not None:
        write_prompt_debug(
            debug_dir=Path(debug_dir),
            debug_name=debug_name,
            prompt=prompt,
            sections=prompt_sections,
        )
    if len(prompt) > MAX_PROMPT_CHARS:
        raise PromptTooLargeError(
            f"TR-generation prompt is still too large after compact fallback ({len(prompt)} chars). "
            f"Reduce LF/examples or repair context."
        )
    raw_response = llm_client.generate(prompt)
    parts = extract_code_sections(raw_response)
    try:
        property_was_normalized = validate_extraction(parts, lf_code=lf_code, allow_property_normalization=allow_property_normalization)
    except Exception:
        if debug_dir is not None:
            _write_debug_artifacts(
                debug_dir=Path(debug_dir),
                debug_name=debug_name,
                prompt=prompt,
                raw_response=raw_response,
                rebeca_code=parts.get("rebeca_code", ""),
                property_code=parts.get("property_code", ""),
            )
        raise
    return TRPackageResult(
        prompt=prompt,
        raw_response=raw_response,
        model_analysis=parts["model_analysis"],
        rebeca_code=parts["rebeca_code"],
        property_code=parts["property_code"],
        expected_outcome=parts["expected_outcome"],
        optional_analysis_model=parts["analysis_variant"] or None,
        optional_analysis_rebeca_code=parts["variant_rebeca_code"] or None,
        optional_analysis_property_code=parts["variant_property_code"] or None,
        provider=llm_client.provider_name,
        model=llm_client.model_name,
        property_was_normalized=property_was_normalized,
    )


def _extract(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def normalize_property_code(property_code: str) -> str:
    stripped = property_code.strip()
    lowered = stripped.lower()
    if not stripped:
        return stripped
    if "property" in lowered and "define" in lowered and "assertion" in lowered:
        return stripped
    expressions: list[str] = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(?:invariant|assert|assertion)\s*\((.*)\)\s*;\s*$", line, re.IGNORECASE)
        if match:
            expressions.append(match.group(1).strip())
    if not expressions:
        return stripped
    define_lines: list[str] = []
    assertion_lines: list[str] = []
    for index, expr in enumerate(expressions, start=1):
        name = f"generated_prop_{index}"
        define_lines.append(f"        {name} = ({expr});")
        assertion_lines.append(f"        P{index}: {name};")
    return "\n".join(["property {", "    define {", *define_lines, "    }", "", "    Assertion {", *assertion_lines, "    }", "}"])


def _candidate_properties_text(candidate_properties: str | dict[str, Any], *, compact: bool) -> str:
    if isinstance(candidate_properties, dict):
        if compact and isinstance(candidate_properties.get("summary"), str):
            return candidate_properties.get("summary", "").strip()
        return json.dumps(candidate_properties, indent=2)
    return candidate_properties.strip()


def summarize_lf_semantics(lf_code: str) -> str:
    analysis = analyze_lf_code(lf_code or "")
    if not analysis.reactors:
        return "No structured LF reactor summary could be extracted. Preserve LF timing semantics conservatively, avoid recurring zero-delay self-loops, and do not invent outputs or recurring behavior."

    lines: list[str] = []
    boundedness_by_reactor: dict[str, list[Any]] = {}
    for item in analysis.state_boundedness():
        boundedness_by_reactor.setdefault(item.reactor_name, []).append(item)
    for reactor in analysis.reactors:
        lines.append(f"Reactor `{reactor.name}`:")
        if reactor.inputs:
            lines.append(f"- inputs: {', '.join(reactor.inputs)}")
            lines.append("- input-triggered reactions should consume stored input values behind presence flags and clear the presence flags after use.")
        if reactor.outputs:
            produced_outputs = reactor.produced_outputs()
            lines.append(f"- declared outputs: {', '.join(reactor.outputs)}")
            for output_name in reactor.outputs:
                if output_name in produced_outputs:
                    lines.append(f"  output `{output_name}` is explicitly produced by LF code.")
                else:
                    lines.append(f"  output `{output_name}` is declared but no explicit `lf_set({output_name}, ...)` was found; do not invent TR sends for it.")
        for timer in reactor.timers:
            if timer.is_periodic:
                lines.append(
                    f"- timer `{timer.name}` is periodic with offset `{timer.offset_text or 'unspecified'}` and period `{timer.period_text}`; recurring self-scheduling must use that positive period, not `after(0)`."
                )
                if (timer.offset_text or "").strip() == "0" and reactor.has_startup_reaction():
                    lines.append(
                        f"  timer `{timer.name}` has zero offset while `{reactor.name}` also has a startup reaction; startup initialization should be handled before timer-driven recurring behavior races with it."
                    )
            else:
                detail = timer.offset_text or "no timing arguments"
                lines.append(f"- timer `{timer.name}` is one-shot or non-periodic with `{detail}`; do not invent recurring self-scheduling.")
        for action in reactor.actions:
            delay = action.min_delay_text or "no declared minimum delay"
            lines.append(f"- {action.kind} action `{action.name}` has declared delay `{delay}`.")
        for reaction in reactor.reactions:
            trigger_text = ", ".join(reaction.triggers) or "no triggers listed"
            source_text = ", ".join(reaction.sources) or "no explicit sources listed"
            effect_text = ", ".join(reaction.effects) or "no explicit effects listed"
            lines.append(f"- reaction triggered by [{trigger_text}] with declared sources [{source_text}] and effects [{effect_text}]")
            if reaction.is_startup:
                lines.append("  this is a startup reaction; do not turn it into invented recurring behavior.")
            if reaction.reads_present:
                lines.append(f"  LF body checks presence for: {', '.join(reaction.reads_present)}; preserve these as `<name>_present` flags.")
            if reaction.reads_values:
                lines.append(f"  LF body reads values for: {', '.join(reaction.reads_values)}; preserve these as stored `<name>_value` or `<name>_val` state.")
            if len(reaction.read_symbols) > 1:
                lines.append("  multiple triggers/sources are involved; preserve separate presence tracking per source and avoid duplicate internal zero-delay reactions with a pending flag.")
            for action_name, delay_text in reaction.scheduled_actions:
                lines.append(
                    f"  reaction body schedules action `{action_name}` with delay `{delay_text}`; preserve that LF delay and do not turn it into autonomous recurring self-triggering unless LF explicitly does so."
                )
            if reaction.set_outputs:
                lines.append(f"  reaction body explicitly produces outputs: {', '.join(reaction.set_outputs)}")
        if reactor.instances:
            lines.append("- contained instances: " + ", ".join(f"{instance.name}:{instance.reactor_class}" for instance in reactor.instances))
        if reactor.connections:
            lines.append(
                "- local connections: "
                + ", ".join(
                    f"{connection.source_instance}.{connection.source_port} -> {connection.target_instance}.{connection.target_port}"
                    for connection in reactor.connections
                )
            )
        for state_info in boundedness_by_reactor.get(reactor.name, []):
            labels = []
            for label in (
                "bounded_by_code",
                "reset_by_code",
                "saturated_by_code",
                "stabilized_by_reachable_branch",
                "potentially_unbounded",
            ):
                if getattr(state_info, label):
                    labels.append(label)
            label_text = ", ".join(labels) if labels else "no recurring write evidence"
            reason_text = "; ".join(state_info.reasons) if state_info.reasons else "No additional reason."
            lines.append(f"- boundedness for state `{state_info.state_name}`: {label_text}. {reason_text}")
            if state_info.recurring_contexts:
                lines.append(f"  recurring contexts: {', '.join(state_info.recurring_contexts)}")
    if analysis.main_instances:
        lines.append("Main reactor instances: " + ", ".join(f"{instance.name}:{instance.reactor_class}" for instance in analysis.main_instances))
    if analysis.main_connections:
        lines.append(
            "Main reactor connections: "
            + ", ".join(
                f"{connection.source_instance}.{connection.source_port} -> {connection.target_instance}.{connection.target_port}"
                for connection in analysis.main_connections
            )
        )
    if analysis.has_feedback_connection_cycle():
        lines.append("The LF connection graph contains a feedback cycle. Preserve it faithfully in the baseline, but do not confuse closed-loop periodic feedback with a mapping bug.")
    if analysis.has_explicit_zero_delay_recurrence():
        lines.append("The LF source appears to contain an explicit zero-delay recurring trigger. Only preserve such a zero-delay loop if it is directly grounded in that LF recurrence, not as an invented timer surrogate.")
    lines.append("If the generated model uses recurring self-scheduling, the delay must be positive unless the LF source clearly models same-tag microstep propagation rather than periodic recurrence.")
    return "\n".join(lines)


def _validate_rebeca_structure(rebeca_code: str) -> None:
    class_specs = _parse_class_specs(rebeca_code)
    if not class_specs:
        raise ValueError("Could not parse any reactiveclass declarations from generated rebeca code.")

    main_match = re.search(r"\bmain\s*\{(.*)\}", rebeca_code, re.DOTALL)
    if not main_match:
        raise ValueError("Generated rebeca code does not contain a parseable main block.")

    instantiation_re = re.compile(r"^\s*(\w+)\s+(\w+)\(([^)]*)\)\s*:\s*\(([^)]*)\)\s*;\s*$")
    errors: list[str] = []
    for raw_line in main_match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        match = instantiation_re.match(line)
        if not match:
            continue
        class_name = match.group(1)
        if class_name not in class_specs:
            continue
        knownrebec_args = _count_args(match.group(3))
        initial_args = _count_args(match.group(4))
        expected_knownrebecs, expected_initial = class_specs[class_name]
        if knownrebec_args != expected_knownrebecs:
            errors.append(
                f"{class_name} instantiation expects {expected_knownrebecs} knownrebec arguments but got {knownrebec_args}: `{line}`"
            )
        if initial_args != expected_initial:
            errors.append(
                f"{class_name} instantiation expects {expected_initial} initial arguments but got {initial_args}: `{line}`"
            )

    if errors:
        raise ValueError("Generated rebeca main block has invalid constructor/init arity:\n" + "\n".join(errors))


def _validate_no_zero_delay_self_loop_cycles(rebeca_code: str, *, lf_code: str = "") -> None:
    analysis = analyze_lf_code(lf_code or "")
    graph = _build_zero_delay_self_call_graph(rebeca_code)
    cycles = _find_graph_cycles(graph)
    if not cycles:
        return
    if analysis.has_explicit_zero_delay_recurrence():
        return
    cycle_lines = [f"- {' -> '.join(cycle)}" for cycle in cycles]
    lf_hint = ""
    if any(timer.is_periodic for reactor in analysis.reactors for timer in reactor.timers):
        lf_hint = "\nThe LF source contains periodic timer-based behavior, so recurring self-scheduling should use the LF timer period instead of `after(0)`."
    raise ValueError(
        "Generated rebeca code contains a recurring zero-delay self-loop or self-loop cycle.\n"
        "This can freeze logical time at 0, grow queues, and break LF-faithful periodic semantics.\n"
        "Recurring zero-delay self-scheduling detected. Use the LF timer period, preserve the LF action delay, or remove the loop if the LF behavior is one-shot.\n"
        f"Detected cycles:\n{chr(10).join(cycle_lines)}{lf_hint}"
    )


def _validate_statevars_initialized(rebeca_code: str) -> None:
    errors: list[str] = []
    for class_name, class_body in _iter_class_bodies(rebeca_code):
        statevars = _parse_rebeca_statevars(class_body)
        if not statevars:
            continue
        constructor_body = _find_constructor_body(class_name, class_body)
        if constructor_body is None:
            errors.append(
                f"{class_name} declares statevars {', '.join(name for name, _ in statevars)} but has no constructor that explicitly initializes them."
            )
            continue
        for state_name, _state_type in statevars:
            if not _is_state_initialized_in_constructor(state_name, constructor_body):
                errors.append(
                    f"{class_name}.{state_name} is declared in statevars but is not explicitly initialized in the constructor."
                )
    if errors:
        raise ValueError(
            "Generated rebeca code contains uninitialized state variables. "
            "Every statevar, including helper variables such as pending/presence flags, must be explicitly initialized in the constructor.\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def _validate_presence_mapping_expectations(rebeca_code: str, *, lf_code: str = "") -> None:
    analysis = analyze_lf_code(lf_code or "")
    if not analysis.reactors:
        return

    lowered_rebeca = rebeca_code.lower()
    errors: list[str] = []
    for reactor in analysis.reactors:
        for reaction in reactor.reactions:
            required_present = set(reaction.reads_present)
            if len(reaction.read_symbols) > 1:
                required_present.update(reaction.read_symbols)
            for symbol in required_present:
                if not re.search(rf"\b{re.escape(symbol.lower())}_[a-z0-9_]*present\b", lowered_rebeca):
                    errors.append(
                        f"LF reaction in `{reactor.name}` uses presence-sensitive symbol `{symbol}`, but no matching `*_present` statevar was found in the generated Rebeca code."
                    )
            for symbol in reaction.reads_values:
                if not re.search(rf"\b{re.escape(symbol.lower())}_[a-z0-9_]*(?:value|val)\b", lowered_rebeca):
                    errors.append(
                        f"LF reaction in `{reactor.name}` reads `{symbol}->value`, but no matching stored `*_value`/`*_val` statevar was found in the generated Rebeca code."
                    )
    if errors:
        raise ValueError(
            "Generated rebeca code does not appear to preserve LF input/action presence or stored values.\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def _validate_pending_guards_for_zero_delay_internal_reactions(rebeca_code: str) -> None:
    errors: list[str] = []
    for class_name, class_body in _iter_class_bodies(rebeca_code):
        statevars = dict(_parse_rebeca_statevars(class_body))
        msgsrv_map = {msg_name: (params, body) for msg_name, params, body in _iter_msgsrv_details(class_body)}
        for source_name, (params, body) in msgsrv_map.items():
            source_has_params = _count_args(params) > 0
            scheduled_targets = re.findall(
                r"self\.(\w+)\s*\([^;{}]*\)\s*after\s*\(\s*0\s*\)\s*;",
                body,
                re.IGNORECASE,
            )
            for target_name in scheduled_targets:
                if target_name == source_name or target_name not in msgsrv_map:
                    continue
                target_params, target_body = msgsrv_map[target_name]
                target_has_params = _count_args(target_params) > 0
                if not source_has_params and target_has_params:
                    continue
                if not source_has_params and not target_has_params:
                    continue
                guard_flag = _extract_pending_guard_flag(body, target_name)
                if guard_flag is None:
                    errors.append(
                        f"{class_name}.{source_name} schedules internal zero-delay self-reaction `{target_name}` without a pending guard."
                    )
                    continue
                state_type = statevars.get(guard_flag, "")
                if state_type.lower() not in {"boolean", "bool"}:
                    errors.append(
                        f"{class_name}.{guard_flag} should be a boolean pending flag guarding `{target_name}`, but no boolean statevar declaration was found."
                    )
                if not re.search(rf"\b{re.escape(guard_flag)}\s*=\s*false\s*;", target_body, re.IGNORECASE):
                    errors.append(
                        f"{class_name}.{target_name} does not reset pending flag `{guard_flag}` to false."
                    )
    if errors:
        raise ValueError(
            "Generated rebeca code contains unsafe zero-delay internal reaction scheduling.\n"
            "Use stored input values plus presence flags, and guard repeated internal self-scheduling with a boolean pending flag.\n"
            + "\n".join(f"- {error}" for error in errors)
        )


def _parse_class_specs(rebeca_code: str) -> dict[str, tuple[int, int]]:
    class_specs: dict[str, tuple[int, int]] = {}
    class_re = re.compile(r"reactiveclass\s+(\w+)\s*\([^)]*\)\s*\{", re.IGNORECASE)
    matches = list(class_re.finditer(rebeca_code))
    for index, match in enumerate(matches):
        class_name = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else rebeca_code.find("main", start)
        if end == -1:
            end = len(rebeca_code)
        body = rebeca_code[start:end]
        knownrebec_count = 0
        known_block = re.search(r"\bknownrebecs\s*\{(.*?)\}", body, re.DOTALL | re.IGNORECASE)
        if known_block:
            knownrebec_count = len(re.findall(r"^\s*\w+\s+\w+\s*;", known_block.group(1), re.MULTILINE))
        initial_count = 0
        initial_match = re.search(r"\bmsgsrv\s+initial\s*\(([^)]*)\)", body, re.IGNORECASE)
        if initial_match:
            initial_count = _count_args(initial_match.group(1))
        class_specs[class_name] = (knownrebec_count, initial_count)
    return class_specs


def _build_zero_delay_self_call_graph(rebeca_code: str) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for class_name, class_body in _iter_class_bodies(rebeca_code):
        msgsrvs = {msg_name: body for msg_name, body in _iter_msgsrv_bodies(class_body)}
        for msg_name, body in msgsrvs.items():
            source = f"{class_name}.{msg_name}"
            graph.setdefault(source, set())
            for target_name in re.findall(r"self\.(\w+)\s*\([^;{}]*\)\s*after\s*\(\s*0\s*\)\s*;", body, re.IGNORECASE):
                if target_name in msgsrvs:
                    graph[source].add(f"{class_name}.{target_name}")
    return graph


def _find_graph_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    seen_cycles: set[tuple[str, ...]] = set()

    def dfs(node: str, path: list[str], visiting: set[str]) -> None:
        visiting.add(node)
        path.append(node)
        for neighbor in graph.get(node, set()):
            if neighbor in visiting:
                cycle = path[path.index(neighbor):] + [neighbor]
                normalized = _normalize_cycle(cycle)
                if normalized not in seen_cycles:
                    seen_cycles.add(normalized)
                    cycles.append(list(normalized))
                continue
            if neighbor not in path:
                dfs(neighbor, path, visiting)
        path.pop()
        visiting.remove(node)

    for node in graph:
        dfs(node, [], set())
    return cycles


def _normalize_cycle(cycle: list[str]) -> tuple[str, ...]:
    core = cycle[:-1]
    if not core:
        return tuple(cycle)
    rotations = [tuple(core[index:] + core[:index] + [core[index]]) for index in range(len(core))]
    return min(rotations)


def _iter_class_bodies(rebeca_code: str) -> list[tuple[str, str]]:
    classes: list[tuple[str, str]] = []
    class_re = re.compile(r"reactiveclass\s+(\w+)\s*\([^)]*\)\s*\{", re.IGNORECASE)
    for match in class_re.finditer(rebeca_code):
        class_name = match.group(1)
        body, _ = _extract_braced_block(rebeca_code, match.end() - 1)
        classes.append((class_name, body))
    return classes


def _iter_msgsrv_bodies(class_body: str) -> list[tuple[str, str]]:
    msgsrvs: list[tuple[str, str]] = []
    msgsrv_re = re.compile(r"msgsrv\s+(\w+)\s*\([^)]*\)\s*\{", re.IGNORECASE)
    for match in msgsrv_re.finditer(class_body):
        msg_name = match.group(1)
        body, _ = _extract_braced_block(class_body, match.end() - 1)
        msgsrvs.append((msg_name, body))
    return msgsrvs


def _iter_msgsrv_details(class_body: str) -> list[tuple[str, str, str]]:
    msgsrvs: list[tuple[str, str, str]] = []
    msgsrv_re = re.compile(r"msgsrv\s+(\w+)\s*\(([^)]*)\)\s*\{", re.IGNORECASE)
    for match in msgsrv_re.finditer(class_body):
        msg_name = match.group(1)
        params = match.group(2)
        body, _ = _extract_braced_block(class_body, match.end() - 1)
        msgsrvs.append((msg_name, params, body))
    return msgsrvs


def _parse_rebeca_statevars(class_body: str) -> list[tuple[str, str]]:
    statevars_block = re.search(r"\bstatevars\s*\{(.*?)\}", class_body, re.DOTALL | re.IGNORECASE)
    if not statevars_block:
        return []
    statevars: list[tuple[str, str]] = []
    for line in statevars_block.group(1).splitlines():
        stripped = line.strip().rstrip(";")
        if not stripped:
            continue
        match = re.match(r"^(.*?)\s+([A-Za-z_]\w*)(?:\s*\[[^\]]+\])?$", stripped)
        if not match:
            continue
        statevars.append((match.group(2), match.group(1).strip()))
    return statevars


def _find_constructor_body(class_name: str, class_body: str) -> str | None:
    constructor_re = re.compile(rf"\b{re.escape(class_name)}\s*\(([^)]*)\)\s*\{{", re.IGNORECASE)
    match = constructor_re.search(class_body)
    if not match:
        return None
    body, _ = _extract_braced_block(class_body, match.end() - 1)
    return body


def _is_state_initialized_in_constructor(state_name: str, constructor_body: str) -> bool:
    return re.search(rf"\b{re.escape(state_name)}\s*=\s*[^=]", constructor_body, re.IGNORECASE) is not None


def _extract_pending_guard_flag(body: str, target_name: str) -> str | None:
    patterns = (
        re.compile(
            rf"if\s*\(\s*!\s*([A-Za-z_]\w*)\s*\)\s*\{{.*?\b\1\s*=\s*true\s*;.*?self\.{re.escape(target_name)}\s*\([^;{{}}]*\)\s*after\s*\(\s*0\s*\)\s*;",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            rf"if\s*\(\s*([A-Za-z_]\w*)\s*==\s*false\s*\)\s*\{{.*?\b\1\s*=\s*true\s*;.*?self\.{re.escape(target_name)}\s*\([^;{{}}]*\)\s*after\s*\(\s*0\s*\)\s*;",
            re.IGNORECASE | re.DOTALL,
        ),
    )
    for pattern in patterns:
        match = pattern.search(body)
        if match:
            return match.group(1)
    return None


def _extract_braced_block(text: str, open_brace_index: int) -> tuple[str, int]:
    depth = 0
    block_chars: list[str] = []
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
            if depth == 1:
                continue
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(block_chars), index
        if depth >= 1:
            block_chars.append(char)
    raise ValueError("Unbalanced braces while parsing generated rebeca code.")


def _count_args(arg_text: str) -> int:
    stripped = arg_text.strip()
    if not stripped:
        return 0
    return len([part for part in stripped.split(",") if part.strip()])


def _write_debug_artifacts(*, debug_dir: Path, debug_name: str, prompt: str, raw_response: str, rebeca_code: str, property_code: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_prompt_debug(
        debug_dir=debug_dir,
        debug_name=debug_name,
        prompt=prompt,
        sections={"full_prompt": prompt},
    )
    (debug_dir / f"{debug_name}.raw_response.txt").write_text(raw_response, encoding="utf-8")
    (debug_dir / f"{debug_name}.rebeca.extracted.txt").write_text(rebeca_code, encoding="utf-8")
    (debug_dir / f"{debug_name}.property.extracted.txt").write_text(property_code, encoding="utf-8")
