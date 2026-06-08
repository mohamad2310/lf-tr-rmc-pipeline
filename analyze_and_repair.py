from __future__ import annotations

import re
from dataclasses import dataclass

from lf_analysis import analyze_lf_code
from prompt_utils import MAX_REPAIR_CONTEXT_CHARS, truncate_text
from run_rmc import RMCResult, looks_like_java_runtime_mismatch

FAILURE_CASES = {
    "success",
    "syntax_or_parser_error",
    "property_parser_error",
    "tool_or_environment_error",
    "terminal_deadlock",
    "deadlock",
    "state_explosion",
    "timeout_or_practically_non_terminating",
    "property_too_strong_or_model_property_mismatch",
}

MAX_COUNTEREXAMPLE_TRANSITIONS = 8
MAX_FAILURE_STATE_CHARS = 1_800
MAX_RMC_SUMMARY_CHARS = 6_000
MAX_REPAIR_REBECA_CHARS = 14_000
MAX_REPAIR_PROPERTY_CHARS = 4_000
MAX_BASELINE_ANALYSIS_CHARS = 4_000


@dataclass
class RepairPlan:
    failure_case: str
    detail_reason: str | None
    summary: str
    repair_prompt: str
    should_repair_baseline: bool


def classify_failure_case(result: RMCResult, baseline_analysis: str = "") -> str:
    text = "\n".join(
        [
            result.translator_log or "",
            result.compile_log or "",
            result.execution_log or "",
            result.output_xml_text or "",
            result.statespace_xml_text or "",
            result.progress_text or "",
            result.notes or "",
            baseline_analysis or "",
            "\n".join(result.warnings or []),
        ]
    ).lower()

    if result.status == "satisfied":
        return "success"

    if result.status == "syntax_error":
        if "property" in text and ("parse" in text or "parser" in text):
            return "property_parser_error"
        return "syntax_or_parser_error"

    if result.status == "tool_error":
        if _looks_like_environment_error(text) or not _has_verification_evidence(result):
            return "tool_or_environment_error"
        if _looks_like_state_explosion(text):
            return "state_explosion"
        return "tool_or_environment_error"

    if result.status == "state_explosion":
        return "state_explosion"

    if result.status == "deadlock":
        if _looks_like_terminal_deadlock(text):
            return "terminal_deadlock"
        return "deadlock"

    if result.status == "timeout":
        if _has_verification_evidence(result) and _looks_like_state_explosion(text):
            return "state_explosion"
        return "timeout_or_practically_non_terminating"

    if result.status == "assertion_failed":
        return "property_too_strong_or_model_property_mismatch"

    return "tool_or_environment_error"


def build_repair_prompt(
    result: RMCResult,
    lf_code: str,
    previous_rebeca: str,
    previous_property: str,
    baseline_analysis: str = "",
) -> RepairPlan:
    failure_case = classify_failure_case(result, baseline_analysis=baseline_analysis)
    if failure_case not in FAILURE_CASES:
        failure_case = "tool_or_environment_error"

    if failure_case == "success":
        return RepairPlan(
            "success",
            None,
            "No repair needed.",
            "No repair needed; the verification result was satisfied.",
            False,
        )

    detail_reason = _classify_detail_reason(
        result=result,
        failure_case=failure_case,
        lf_code=lf_code,
        previous_rebeca=previous_rebeca,
    )
    should_repair_baseline = _should_repair_baseline(failure_case, detail_reason)
    summary = _summary_for_case(failure_case)
    extra_guidance = _extra_guidance_for_case(failure_case, detail_reason=detail_reason)
    lf_summary = _summarize_lf_structure(lf_code)
    compact_rmc_summary = truncate_text(
        _format_rmc_failure_summary(result, failure_case),
        MAX_RMC_SUMMARY_CHARS,
    )
    rebeca_context = truncate_text(previous_rebeca.strip() or "None", MAX_REPAIR_REBECA_CHARS)
    property_context = truncate_text(previous_property.strip() or "None", MAX_REPAIR_PROPERTY_CHARS)
    analysis_context = truncate_text(baseline_analysis.strip() or "None", MAX_BASELINE_ANALYSIS_CHARS)

    prompt = _compose_repair_prompt(
        failure_case=failure_case,
        summary=summary,
        extra_guidance=extra_guidance,
        lf_summary=lf_summary,
        previous_rebeca=rebeca_context,
        previous_property=property_context,
        baseline_analysis=analysis_context,
        rmc_status=result.status,
        compact_rmc_summary=compact_rmc_summary,
        detail_reason=detail_reason,
        should_repair_baseline=should_repair_baseline,
    )
    if len(prompt) > MAX_REPAIR_CONTEXT_CHARS:
        prompt = _compose_compact_repair_prompt(
            failure_case=failure_case,
            summary=summary,
            extra_guidance=extra_guidance,
            lf_summary=truncate_text(lf_summary, 3_000),
            previous_rebeca=truncate_text(rebeca_context, 8_000),
            previous_property=truncate_text(property_context, 2_000),
            baseline_analysis=truncate_text(analysis_context, 2_000),
            rmc_status=result.status,
            compact_rmc_summary=truncate_text(compact_rmc_summary, 3_500),
            detail_reason=detail_reason,
            should_repair_baseline=should_repair_baseline,
        )
    prompt = truncate_text(prompt, MAX_REPAIR_CONTEXT_CHARS)

    return RepairPlan(
        failure_case=failure_case,
        detail_reason=detail_reason,
        summary=summary,
        repair_prompt=prompt,
        should_repair_baseline=should_repair_baseline,
    )


def _compose_repair_prompt(
    *,
    failure_case: str,
    summary: str,
    extra_guidance: str,
    lf_summary: str,
    previous_rebeca: str,
    previous_property: str,
    baseline_analysis: str,
    rmc_status: str,
    compact_rmc_summary: str,
    detail_reason: str | None,
    should_repair_baseline: bool,
) -> str:
    return f"""You are preparing the next Timed Rebeca regeneration round after an RMC run.

Workflow constraints:
1. Diagnose in this order only: syntax/environment, structural formatting, initialization, ongoing triggers or periodic behavior, unbounded variable growth, unreachable branches or broken guards, model/property mismatch, and only then property wrongness.
2. Preserve the strict LF-faithful baseline semantics unless you explicitly label an analysis-oriented variant.
3. If the deadlock is terminal deadlock from normal completion, do not add artificial infinite behavior.
4. If verification timed out and the model is periodic with unbounded growth, treat it as state explosion.
5. If a duplicated or inconsistent guard breaks a control branch, say that clearly.
6. If the property is stronger than the model supports, weaken the property before changing the model.
7. Do not silently change baseline semantics just to make the checker terminate.
8. Do not use the full RMC XML or raw logs as context. Work only from the compact failure summary below.

Case-specific guidance:
{extra_guidance}

Raw RMC status:
{rmc_status}

Normalized failure classification:
{failure_case}

Detailed reason:
{detail_reason or 'none'}

Strict baseline repair allowed:
{should_repair_baseline}

Summary:
{summary}

Compact RMC failure summary:
{compact_rmc_summary}

LF structural summary:
{lf_summary}

Current generated baseline .rebeca:
{previous_rebeca}

Current generated baseline .property:
{previous_property}

Previous baseline model analysis:
{baseline_analysis}

Return:
- a concise diagnosis,
- the exact repair strategy,
- exactly one fenced `rebeca` block and one fenced `property` block if a corrected baseline package is justified,
- otherwise a clear explanation of why the baseline should be kept and the result should be reinterpreted.
"""


def _compose_compact_repair_prompt(
    *,
    failure_case: str,
    summary: str,
    extra_guidance: str,
    lf_summary: str,
    previous_rebeca: str,
    previous_property: str,
    baseline_analysis: str,
    rmc_status: str,
    compact_rmc_summary: str,
    detail_reason: str | None,
    should_repair_baseline: bool,
) -> str:
    return f"""You are preparing a compact repair context for the next Timed Rebeca regeneration round.

Use only the compact facts below. Do not infer behavior from missing log details.

Raw RMC status: {rmc_status}
Normalized failure classification: {failure_case}
Detailed reason: {detail_reason or 'none'}
Strict baseline repair allowed: {should_repair_baseline}
Summary: {summary}

Short repair instructions:
{extra_guidance}

Compact RMC failure summary:
{compact_rmc_summary}

LF structural summary:
{lf_summary}

Current generated baseline .rebeca:
{previous_rebeca}

Current generated baseline .property:
{previous_property}

Previous baseline model analysis:
{baseline_analysis}

Return:
- a concise diagnosis,
- the exact repair strategy,
- exactly one fenced `rebeca` block and one fenced `property` block if a corrected baseline package is justified,
- otherwise a clear explanation of why the baseline should be kept and the result should be reinterpreted.
"""


def _summarize_lf_structure(lf_code: str) -> str:
    analysis = analyze_lf_code(lf_code or "")
    if not analysis.reactors:
        return "No structured LF reactor summary could be extracted."

    lines: list[str] = []
    for reactor in analysis.reactors:
        lines.append(f"Reactor `{reactor.name}`:")
        if reactor.inputs:
            lines.append(f"- inputs: {', '.join(reactor.inputs)}")
        if reactor.outputs:
            lines.append(f"- outputs: {', '.join(reactor.outputs)}")
        if reactor.timers:
            timer_bits = []
            for timer in reactor.timers:
                part = timer.name
                if timer.offset_text:
                    part += f" offset={timer.offset_text}"
                if timer.period_text:
                    part += f" period={timer.period_text}"
                timer_bits.append(part)
            lines.append(f"- timers: {', '.join(timer_bits)}")
        if reactor.actions:
            action_bits = []
            for action in reactor.actions:
                part = action.name
                if action.min_delay_text:
                    part += f" min_delay={action.min_delay_text}"
                part += f" kind={action.kind}"
                action_bits.append(part)
            lines.append(f"- actions: {', '.join(action_bits)}")
        if reactor.states:
            init_bits = []
            for state in reactor.states:
                status = "initialized" if state.is_initialized else "uninitialized"
                init_bits.append(f"{state.name} ({status})")
            lines.append(f"- states: {', '.join(init_bits)}")
        if reactor.reactions:
            trigger_bits = []
            for reaction in reactor.reactions:
                triggers = ", ".join(reaction.triggers) if reaction.triggers else "no explicit trigger"
                trigger_bits.append(triggers)
            lines.append(f"- reactions: {'; '.join(trigger_bits)}")
        scheduled_actions = [
            f"{action_name} delay={delay_text}"
            for reaction in reactor.reactions
            for action_name, delay_text in reaction.scheduled_actions
        ]
        if scheduled_actions:
            schedule_bits = scheduled_actions
            lines.append(f"- scheduled actions: {', '.join(schedule_bits)}")
        produced_outputs = reactor.produced_outputs()
        if produced_outputs:
            lines.append(f"- produced outputs: {', '.join(sorted(produced_outputs))}")
        unproduced_outputs = [name for name in reactor.outputs if name not in produced_outputs]
        if unproduced_outputs:
            lines.append(f"- declared but not obviously produced outputs: {', '.join(unproduced_outputs)}")
    if analysis.main_instances:
        lines.append("Main instances: " + ", ".join(f"{instance.name}:{instance.reactor_class}" for instance in analysis.main_instances))
    if analysis.main_connections:
        lines.append(
            "Main connections: "
            + ", ".join(
                f"{connection.source_instance}.{connection.source_port} -> {connection.target_instance}.{connection.target_port}"
                for connection in analysis.main_connections
            )
        )
    if analysis.has_feedback_connection_cycle():
        lines.append("The LF connection graph contains a feedback cycle.")
    return "\n".join(lines)


def _format_rmc_failure_summary(result: RMCResult, failure_case: str) -> str:
    text_sources = [
        result.output_xml_text or "",
        result.execution_log or "",
        result.progress_text or "",
        result.statespace_xml_text or "",
        result.translator_log or "",
        result.compile_log or "",
    ]
    combined = "\n".join(text_sources)
    lower = combined.lower()

    property_result = _first_match(
        combined,
        (
            r"<checked-property>.*?<result>\s*([^<]+?)\s*</result>",
            r"<result>\s*([^<]+?)\s*</result>",
        ),
    )
    property_message = _first_match(
        combined,
        (
            r"<checked-property>.*?<message>\s*([^<]+?)\s*</message>",
            r"<message>\s*([^<]+?)\s*</message>",
        ),
    )
    reached_states = _first_int(
        combined,
        (
            r"<reached-states>\s*(\d+)\s*</reached-states>",
            r"reached[- ]states[^0-9]*(\d+)",
        ),
    )
    reached_transitions = _first_int(
        combined,
        (
            r"<reached-transitions>\s*(\d+)\s*</reached-transitions>",
            r"reached[- ]transitions[^0-9]*(\d+)",
        ),
    )
    consumed_memory = _first_int(
        combined,
        (
            r"<consumed-mem>\s*(\d+)\s*</consumed-mem>",
            r"consumed[- ]mem[^0-9]*(\d+)",
            r"memory[^0-9]*(\d+)",
        ),
    )

    queue_overflow = "queue overflow" in lower
    assertion_failed = "assertion failed" in lower
    deadlock = "deadlock" in lower
    transition_source = "\n".join(
        [
            result.execution_log or "",
            result.output_xml_text or "",
            result.statespace_xml_text or "",
        ]
    )
    final_state = (
        _extract_last_counterexample_state(result.execution_log or "")
        or _extract_last_counterexample_state(result.output_xml_text or "")
        or _extract_last_counterexample_state(result.statespace_xml_text or "")
    )
    max_queue = _largest_queue_from_state(final_state or "")
    transitions = _extract_counterexample_transitions(transition_source)

    lines = [
        f"- raw_rmc_status: {result.status}",
        f"- normalized_failure_case: {failure_case}",
        f"- property_result: {property_result or 'unknown'}",
        f"- property_message: {property_message or 'none'}",
        f"- reached_states: {reached_states if reached_states is not None else 'unknown'}",
        f"- reached_transitions: {reached_transitions if reached_transitions is not None else 'unknown'}",
        f"- consumed_memory: {consumed_memory if consumed_memory is not None else 'unknown'}",
        f"- queue_overflow: {queue_overflow}",
        f"- assertion_failed: {assertion_failed}",
        f"- deadlock_observed: {deadlock}",
    ]
    if max_queue:
        lines.append(f"- largest_detected_queue_rebec: {max_queue['rebec_name']}")
        lines.append(f"- largest_detected_queue_size: {max_queue['queue_size']}")
    if transitions:
        lines.append("- first_counterexample_transitions:")
        lines.extend(f"  {idx}. {step}" for idx, step in enumerate(transitions, start=1))
    if final_state:
        lines.append("- final_failure_state_excerpt:")
        lines.append(truncate_text(final_state.strip(), MAX_FAILURE_STATE_CHARS))
    warnings = [warning for warning in result.warnings or [] if warning.strip()]
    if warnings:
        lines.append("- warnings:")
        lines.extend(f"  - {warning}" for warning in warnings[:5])
    if result.notes:
        lines.append(f"- notes: {result.notes}")
    return "\n".join(lines)


def _classify_detail_reason(
    *,
    result: RMCResult,
    failure_case: str,
    lf_code: str,
    previous_rebeca: str,
) -> str | None:
    if failure_case == "tool_or_environment_error":
        text = "\n".join(
            [
                result.translator_log or "",
                result.compile_log or "",
                result.execution_log or "",
                result.output_xml_text or "",
                result.progress_text or "",
                result.notes or "",
                "\n".join(result.warnings or []),
            ]
        ).lower()
        if looks_like_java_runtime_mismatch(text) or "a jni error has occurred" in text:
            return "java_runtime_mismatch"
        return "tool_or_environment_error"

    if failure_case not in {"state_explosion", "timeout_or_practically_non_terminating"}:
        return None

    analysis = analyze_lf_code(lf_code or "")
    text = "\n".join(
        [
            result.execution_log or "",
            result.output_xml_text or "",
            result.statespace_xml_text or "",
            result.progress_text or "",
            result.notes or "",
        ]
    ).lower()

    if _looks_like_timer_mapping_error(analysis, previous_rebeca, text):
        return "timer_mapping_error"
    if _looks_like_presence_mapping_error(analysis, previous_rebeca):
        return "presence_mapping_error"
    if any(item.potentially_unbounded for item in analysis.state_boundedness()):
        return "periodic_unbounded_state"
    if analysis.has_feedback_connection_cycle():
        return "closed_loop_feedback"
    if any(timer.is_periodic for reactor in analysis.reactors for timer in reactor.timers):
        return "bounded_periodic_interleavings"
    return "tool_or_state_space_limitation"


def _should_repair_baseline(failure_case: str, detail_reason: str | None) -> bool:
    if failure_case in {
        "syntax_or_parser_error",
        "property_parser_error",
        "deadlock",
        "property_too_strong_or_model_property_mismatch",
    }:
        return True
    if failure_case == "tool_or_environment_error":
        return False
    if failure_case not in {"state_explosion", "timeout_or_practically_non_terminating"}:
        return True
    return detail_reason in {"timer_mapping_error", "presence_mapping_error"}


def _looks_like_timer_mapping_error(analysis, previous_rebeca: str, text: str) -> bool:
    if not any(timer.is_periodic for reactor in analysis.reactors for timer in reactor.timers):
        return False
    if not re.search(r"self\.(\w+)\s*\([^;{}]*\)\s*after\s*\(\s*0\s*\)\s*;", previous_rebeca, re.IGNORECASE):
        return False
    repeated_same_time = any(
        token in text
        for token in (
            "<now>0</now>",
            'executiontime="0"',
            'shift="0"',
            "queue overflow",
        )
    )
    return repeated_same_time


def _looks_like_presence_mapping_error(analysis, previous_rebeca: str) -> bool:
    lowered_rebeca = previous_rebeca.lower()
    for reactor in analysis.reactors:
        for reaction in reactor.reactions:
            required_present = set(reaction.reads_present)
            if len(reaction.read_symbols) > 1:
                required_present.update(reaction.read_symbols)
            for symbol in required_present:
                if not re.search(rf"\b{re.escape(symbol.lower())}_[a-z0-9_]*present\b", lowered_rebeca):
                    return True
            for symbol in reaction.reads_values:
                if not re.search(rf"\b{re.escape(symbol.lower())}_[a-z0-9_]*(?:value|val)\b", lowered_rebeca):
                    return True
            if len(reaction.read_symbols) > 1 and "pending" not in lowered_rebeca:
                return True
    return False


def _extract_counterexample_transitions(text: str) -> list[str]:
    steps: list[str] = []
    pattern = re.compile(
        r"<transition[^>]*?>\s*<messageserver\b[^>]*sender=\"([^\"]+)\"[^>]*owner=\"([^\"]+)\"[^>]*title=\"([^\"]+)\"[^>]*/>\s*</transition>",
        re.IGNORECASE | re.DOTALL,
    )
    for sender, owner, title in pattern.findall(text):
        steps.append(f"sender={sender}, owner={owner}, msgsrv={title}")
        if len(steps) >= MAX_COUNTEREXAMPLE_TRANSITIONS:
            break
    return steps


def _extract_last_counterexample_state(text: str) -> str | None:
    states = re.findall(r"(<state\b.*?</state>)", text, flags=re.IGNORECASE | re.DOTALL)
    if not states:
        return None
    return states[-1]


def _largest_queue_from_state(state_block: str) -> dict[str, int | str] | None:
    if not state_block.strip():
        return None
    best: dict[str, int | str] | None = None
    for rebec_name, rebec_body in re.findall(
        r"<rebec\s+name=\"([^\"]+)\"[^>]*>(.*?)</rebec>",
        state_block,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        queue_match = re.search(r"<queue>(.*?)</queue>", rebec_body, flags=re.IGNORECASE | re.DOTALL)
        if not queue_match:
            continue
        queue_size = len(re.findall(r"<message\b", queue_match.group(1), flags=re.IGNORECASE))
        if best is None or queue_size > int(best["queue_size"]):
            best = {"rebec_name": rebec_name, "queue_size": queue_size}
    return best


def _first_match(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def _first_int(text: str, patterns: tuple[str, ...]) -> int | None:
    match = _first_match(text, patterns)
    if not match:
        return None
    try:
        return int(match)
    except ValueError:
        return None


def _looks_like_terminal_deadlock(text: str) -> bool:
    return any(
        token in text
        for token in (
            "terminal deadlock",
            "naturally terminate",
            "natural termination",
            "one-shot",
            "terminates after",
            "terminating baseline",
            "natural quiescence",
            "no future scheduled events",
            "no messages remain",
        )
    )


def _looks_like_state_explosion(text: str) -> bool:
    if any(
        token in text
        for token in (
            "state explosion",
            "queue overflow",
            "out of memory",
            "memory exhausted",
            "practically non-terminating",
            "keeps growing",
            "exploding state space",
        )
    ):
        return True
    if ("statespace" in text or "state space" in text) and any(
        token in text
        for token in (
            "reached-states",
            "reached states",
            "reached-transitions",
            "reached transitions",
            "consumed-mem",
            "consumed mem",
            "queue overflow",
        )
    ):
        return True
    return False


def _has_verification_evidence(result: RMCResult) -> bool:
    return any(
        bool((field or "").strip())
        for field in (
            result.execution_log,
            result.output_xml_text,
            result.statespace_xml_text,
            result.progress_text,
        )
    )


def _looks_like_environment_error(text: str) -> bool:
    if looks_like_java_runtime_mismatch(text):
        return True
    return any(
        token in text
        for token in (
            "a jni error has occurred",
            "java was not found in path",
            "g++ was not found in path",
            "rmc jar not found",
            "could not be compiled by a more recent version of the java runtime",
            "current java runtime is too old",
            "failed to launch under the current java runtime",
        )
    )


def _summary_for_case(failure_case: str) -> str:
    summaries = {
        "syntax_or_parser_error": "RMC most likely rejected model or property syntax/formatting.",
        "property_parser_error": "The property file syntax likely does not match the RMC/Rebeca property parser expectations.",
        "tool_or_environment_error": "The Java, g++, or RMC toolchain failed independently of semantics.",
        "terminal_deadlock": (
            "The baseline most likely reaches natural quiescence. "
            "This looks more like normal completion than a coordination bug."
        ),
        "deadlock": "The model appears to contain a real coordination or control-flow deadlock that needs diagnosis.",
        "state_explosion": "The timeout most likely comes from periodic/unbounded behavior or exploding state space.",
        "timeout_or_practically_non_terminating": "The run exceeded the timeout without enough evidence to call it state explosion confidently.",
        "property_too_strong_or_model_property_mismatch": "The current property set is stronger than the model supports or mismatched with the chosen abstraction.",
    }
    return summaries[failure_case]


def _extra_guidance_for_case(failure_case: str, *, detail_reason: str | None = None) -> str:
    if failure_case == "tool_or_environment_error":
        if detail_reason == "java_runtime_mismatch":
            return (
                "Do not regenerate the baseline model. The current Java runtime cannot launch the RMC jar. "
                "Install a newer Java runtime compatible with this jar and rerun the same strict baseline."
            )
        return (
            "Do not regenerate the baseline model until the external toolchain issue is fixed. "
            "Verify the Java runtime, RMC jar, g++, and local installation first."
        )

    if failure_case == "terminal_deadlock":
        return (
            "Treat this as likely natural completion first. "
            "Do NOT add artificial keep-alive behavior, periodic self-triggers, or environment noise just to avoid deadlock. "
            "Preserve the LF-faithful baseline. "
            "First consider whether the deadlock should be reinterpreted as terminal quiescence and whether only the property set or result interpretation should change. "
            "Only propose a second analysis-oriented non-terminating variant if it is clearly labeled as separate from the baseline."
        )

    if failure_case == "deadlock":
        return (
            "Do not assume terminal quiescence. "
            "Inspect message flow, pending queues, triggers, guard reachability, startup sequencing, and whether any expected reaction is never enabled. "
            "Prefer the smallest semantically justified correction. "
            "Do not add an infinite loop unless the LF semantics already justify recurring behavior."
        )

    if failure_case == "property_too_strong_or_model_property_mismatch":
        return "Prefer weakening or replacing fragile properties before changing the baseline model."

    if failure_case in {"state_explosion", "timeout_or_practically_non_terminating"}:
        if detail_reason == "timer_mapping_error":
            return (
                "This looks like a timer-mapping problem. Repair the strict baseline by preserving LF timer offset/period faithfully. "
                "Recurring timers must not self-schedule with `after(0)`."
            )
        if detail_reason == "presence_mapping_error":
            return (
                "This looks like a presence/value mapping problem. Repair the strict baseline by preserving LF `is_present`/`value` semantics and pending-flag behavior faithfully."
            )
        if detail_reason in {
            "periodic_unbounded_state",
            "closed_loop_feedback",
            "bounded_periodic_interleavings",
            "tool_or_state_space_limitation",
        }:
            return (
                "Do not change the strict baseline semantics to force finite-state verification. "
                "Document the limitation clearly and, if useful, provide only a separate analysis-oriented bounded variant labeled as non-baseline."
            )
        return (
            "Inspect the generated Rebeca code for recurring zero-delay self scheduling, duplicate internal reaction scheduling, "
            "unbounded message accumulation, missing pending flags, and missing explicit initialization before changing property syntax."
        )

    return "Use the general workflow constraints above."
