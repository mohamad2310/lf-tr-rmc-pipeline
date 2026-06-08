from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class LFState:
    name: str
    type_text: str
    initializer_text: str | None = None
    assignment_expressions: list[str] = field(default_factory=list)
    update_expressions: list[str] = field(default_factory=list)

    @property
    def is_initialized(self) -> bool:
        return self.initializer_text is not None


@dataclass
class LFTimer:
    name: str
    offset_text: str = ""
    period_text: str = ""

    @property
    def is_periodic(self) -> bool:
        return bool(self.period_text.strip())

    @property
    def is_zero_period(self) -> bool:
        return self.is_periodic and _is_zero_delay_text(self.period_text)


@dataclass
class LFAction:
    kind: str
    name: str
    min_delay_text: str = ""


@dataclass
class LFReaction:
    triggers: list[str]
    sources: list[str]
    effects: list[str]
    body: str
    scheduled_actions: list[tuple[str, str]]
    set_outputs: list[str]
    reads_present: list[str]
    reads_values: list[str]

    @property
    def is_startup(self) -> bool:
        return any(_normalize_symbol_name(trigger) == "startup" for trigger in self.triggers)

    @property
    def read_symbols(self) -> list[str]:
        return _unique_preserve_order(self.sources + self.reads_present + self.reads_values)


@dataclass
class LFInstance:
    name: str
    reactor_class: str


@dataclass
class LFConnection:
    source_instance: str
    source_port: str
    target_instance: str
    target_port: str


@dataclass
class LFStateBoundedness:
    reactor_name: str
    state_name: str
    bounded_by_code: bool = False
    reset_by_code: bool = False
    saturated_by_code: bool = False
    stabilized_by_reachable_branch: bool = False
    potentially_unbounded: bool = False
    recurring_contexts: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class LFReactor:
    name: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    states: list[LFState] = field(default_factory=list)
    timers: list[LFTimer] = field(default_factory=list)
    actions: list[LFAction] = field(default_factory=list)
    reactions: list[LFReaction] = field(default_factory=list)
    instances: list[LFInstance] = field(default_factory=list)
    connections: list[LFConnection] = field(default_factory=list)

    def state_map(self) -> dict[str, LFState]:
        return {state.name: state for state in self.states}

    def produced_outputs(self) -> set[str]:
        produced: set[str] = set()
        for reaction in self.reactions:
            produced.update(reaction.set_outputs)
        return produced

    def has_startup_reaction(self) -> bool:
        return any(reaction.is_startup for reaction in self.reactions)


@dataclass
class LFAnalysis:
    reactors: list[LFReactor]
    main_instances: list[LFInstance] = field(default_factory=list)
    main_connections: list[LFConnection] = field(default_factory=list)


    def uninitialized_state_names(self) -> set[str]:
        return {state.name for reactor in self.reactors for state in reactor.states if not state.is_initialized}

    def all_connections(self) -> list[LFConnection]:
        combined: list[LFConnection] = []
        for reactor in self.reactors:
            combined.extend(reactor.connections)
        combined.extend(self.main_connections)
        return combined

    def has_feedback_connection_cycle(self) -> bool:
        graph: dict[str, set[str]] = {}
        for connection in self.all_connections():
            source = connection.source_instance or "self"
            target = connection.target_instance or "self"
            graph.setdefault(source, set()).add(target)
        visited: set[str] = set()
        visiting: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            visiting.add(node)
            for neighbor in graph.get(node, set()):
                if neighbor in visiting:
                    return True
                if neighbor not in visited and dfs(neighbor):
                    return True
            visiting.remove(node)
            return False

        return any(dfs(node) for node in graph if node not in visited)

    def recurring_action_names(self, reactor: LFReactor) -> set[str]:
        periodic_timers = {timer.name for timer in reactor.timers if timer.is_periodic}
        recurring_actions: set[str] = set()
        changed = True
        while changed:
            changed = False
            for reaction in reactor.reactions:
                trigger_names = {_normalize_symbol_name(trigger) for trigger in reaction.triggers}
                if trigger_names.intersection(periodic_timers | recurring_actions):
                    for action_name, _delay_text in reaction.scheduled_actions:
                        normalized = _normalize_symbol_name(action_name)
                        if normalized not in recurring_actions:
                            recurring_actions.add(normalized)
                            changed = True
        return recurring_actions

    def recurring_reactions(self, reactor: LFReactor) -> list[LFReaction]:
        periodic_timers = {timer.name for timer in reactor.timers if timer.is_periodic}
        recurring_actions = self.recurring_action_names(reactor)
        recurring: list[LFReaction] = []
        for reaction in reactor.reactions:
            trigger_names = {_normalize_symbol_name(trigger) for trigger in reaction.triggers}
            if trigger_names.intersection(periodic_timers | recurring_actions):
                recurring.append(reaction)
        return recurring

    def has_explicit_zero_delay_recurrence(self) -> bool:
        for reactor in self.reactors:
            trigger_names = {timer.name for timer in reactor.timers if timer.is_zero_period}
            recurring_actions = self.recurring_action_names(reactor)
            for reaction in reactor.reactions:
                recurring_zero = {
                    _normalize_symbol_name(action_name)
                    for action_name, delay_text in reaction.scheduled_actions
                    if _is_zero_delay_text(delay_text)
                }
                for trigger_name in reaction.triggers:
                    normalized_trigger = _normalize_symbol_name(trigger_name)
                    if normalized_trigger in recurring_zero and normalized_trigger in recurring_actions:
                        return True
                if any(_normalize_symbol_name(trigger) in trigger_names for trigger in reaction.triggers):
                    return True
        return False

    def state_boundedness(self) -> list[LFStateBoundedness]:
        results: list[LFStateBoundedness] = []
        for reactor in self.reactors:
            recurring_reactions = self.recurring_reactions(reactor)
            for state in reactor.states:
                writes = [
                    (reaction, *_extract_state_writes(state.name, reaction.body))
                    for reaction in recurring_reactions
                ]
                writes = [entry for entry in writes if entry[1] or entry[2]]
                if not writes:
                    results.append(
                        LFStateBoundedness(
                            reactor_name=reactor.name,
                            state_name=state.name,
                            reasons=["No recurring writes to this state were detected."],
                        )
                    )
                    continue

                assignment_exprs = [expr for _reaction, assignments, _updates in writes for expr in assignments]
                update_exprs = [expr for _reaction, _assignments, updates in writes for expr in updates]
                recurring_contexts = [
                    ", ".join(reaction.triggers) or "unknown recurring trigger"
                    for reaction, _assignments, _updates in writes
                ]

                bounded_by_code = _is_boolean_domain(state.type_text) or (
                    bool(assignment_exprs)
                    and all(_looks_like_finite_assignment(expr) for expr in assignment_exprs)
                    and not update_exprs
                )
                reset_by_code = any(
                    _looks_like_reset_expression(expr, state.initializer_text) for expr in assignment_exprs
                )
                saturated_by_code = any(
                    _looks_like_saturating_expression(expr, state.name) for expr in assignment_exprs + update_exprs
                )
                stabilized_by_reachable_branch = any(
                    _has_state_based_stabilizing_branch(state.name, reaction.body) for reaction, _a, _u in writes
                )
                potentially_unbounded = bool(writes) and not any(
                    [bounded_by_code, reset_by_code, saturated_by_code, stabilized_by_reachable_branch]
                )

                reasons: list[str] = []
                if bounded_by_code:
                    reasons.append("Assignments stay within a finite or boolean domain.")
                if reset_by_code:
                    reasons.append("A recurring reaction assigns an explicit reset/constant value.")
                if saturated_by_code:
                    reasons.append("A recurring update uses an explicit saturating/clamping expression.")
                if stabilized_by_reachable_branch:
                    reasons.append("A recurring reaction contains a state-based stabilizing branch.")
                if potentially_unbounded:
                    reasons.append("Recurring writes were detected without an explicit bound, reset, saturation, or stabilizing branch.")

                results.append(
                    LFStateBoundedness(
                        reactor_name=reactor.name,
                        state_name=state.name,
                        bounded_by_code=bounded_by_code,
                        reset_by_code=reset_by_code,
                        saturated_by_code=saturated_by_code,
                        stabilized_by_reachable_branch=stabilized_by_reachable_branch,
                        potentially_unbounded=potentially_unbounded,
                        recurring_contexts=_unique_preserve_order(recurring_contexts),
                        reasons=reasons,
                    )
                )
        return results


def analyze_lf_code(lf_code: str) -> LFAnalysis:
    original_code = lf_code or ""
    stripped_code = _strip_c_bodies(original_code)
    reactors: list[LFReactor] = []
    main_instances: list[LFInstance] = []
    main_connections: list[LFConnection] = []

    for reactor_name, original_body, stripped_body, is_main in _iter_reactor_blocks(original_code, stripped_code):
        if is_main:
            main_instances = _find_instances(stripped_body)
            main_connections = _find_connections(stripped_body)
            continue

        reactor = LFReactor(name=reactor_name)
        reactor.inputs = _find_port_names(stripped_body, "input")
        reactor.outputs = _find_port_names(stripped_body, "output")
        reactor.states = _find_states(stripped_body)
        reactor.timers = _find_timers(stripped_body)
        reactor.actions = _find_actions(stripped_body)
        reactor.reactions = _find_reactions(original_body)
        reactor.instances = _find_instances(stripped_body)
        reactor.connections = _find_connections(stripped_body)
        _populate_state_writes(reactor)
        reactors.append(reactor)

    return LFAnalysis(
        reactors=reactors,
        main_instances=main_instances,
        main_connections=main_connections,
    )


def _find_port_names(body: str, direction: str) -> list[str]:
    pattern = re.compile(rf"\b{direction}\s+([A-Za-z_]\w*)\b")
    return [match.group(1) for match in pattern.finditer(body)]


def _find_states(body: str) -> list[LFState]:
    states: list[LFState] = []
    state_re = re.compile(r"\bstate\s+([A-Za-z_]\w*)\s*:\s*([^;{}\n]+?)\s*;")
    for match in state_re.finditer(body):
        name = match.group(1)
        type_text = match.group(2).strip()
        initializer = _extract_state_initializer(type_text)
        states.append(
            LFState(
                name=name,
                type_text=type_text,
                initializer_text=initializer,
            )
        )
    return states


def _find_timers(body: str) -> list[LFTimer]:
    timers: list[LFTimer] = []
    timer_re = re.compile(r"\btimer\s+([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?")
    for match in timer_re.finditer(body):
        args = [part.strip() for part in (match.group(2) or "").split(",") if part.strip()]
        offset_text = args[0] if args else ""
        period_text = args[1] if len(args) > 1 else ""
        timers.append(LFTimer(name=match.group(1), offset_text=offset_text, period_text=period_text))
    return timers


def _find_actions(body: str) -> list[LFAction]:
    actions: list[LFAction] = []
    action_re = re.compile(r"\b(logical|physical)\s+action\s+([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?")
    for match in action_re.finditer(body):
        actions.append(
            LFAction(
                kind=match.group(1),
                name=match.group(2),
                min_delay_text=(match.group(3) or "").strip(),
            )
        )
    return actions


def _find_reactions(body: str) -> list[LFReaction]:
    reactions: list[LFReaction] = []
    reaction_re = re.compile(
        r"reaction\s*\((.*?)\)\s*(.*?)\{=(.*?)=\}",
        re.DOTALL | re.IGNORECASE,
    )
    for match in reaction_re.finditer(body):
        triggers = _split_csv(match.group(1))
        header_tail = " ".join(match.group(2).strip().split())
        sources: list[str] = []
        effects: list[str] = []
        if "->" in header_tail:
            left, right = header_tail.split("->", 1)
            sources = _split_csv(left)
            effects = _split_csv(right)
        else:
            sources = _split_csv(header_tail)

        reaction_body = match.group(3).strip()
        scheduled_actions = [
            (schedule_match.group(1), schedule_match.group(2).strip())
            for schedule_match in re.finditer(
                r"\blf_schedule\s*\(\s*([A-Za-z_]\w*)\s*,\s*([^)]+?)\s*\)",
                reaction_body,
                re.IGNORECASE,
            )
        ]
        set_outputs = [
            set_match.group(1)
            for set_match in re.finditer(
                r"\blf_set\s*\(\s*([A-Za-z_]\w*)\s*,",
                reaction_body,
                re.IGNORECASE,
            )
        ]
        reads_present = [
            present_match.group(1)
            for present_match in re.finditer(
                r"\b([A-Za-z_]\w*)\s*->\s*is_present\b",
                reaction_body,
                re.IGNORECASE,
            )
        ]
        reads_values = [
            value_match.group(1)
            for value_match in re.finditer(
                r"\b([A-Za-z_]\w*)\s*->\s*value\b",
                reaction_body,
                re.IGNORECASE,
            )
        ]
        reactions.append(
            LFReaction(
                triggers=triggers,
                sources=sources,
                effects=effects,
                body=reaction_body,
                scheduled_actions=scheduled_actions,
                set_outputs=set_outputs,
                reads_present=_unique_preserve_order(reads_present),
                reads_values=_unique_preserve_order(reads_values),
            )
        )
    return reactions


def _find_instances(body: str) -> list[LFInstance]:
    instances: list[LFInstance] = []
    instance_re = re.compile(
        r"\b([A-Za-z_]\w*)\s*=\s*new\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*;",
        re.DOTALL,
    )
    for match in instance_re.finditer(body):
        instances.append(
            LFInstance(
                name=match.group(1).strip(),
                reactor_class=match.group(2).strip(),
            )
        )
    return instances


def _find_connections(body: str) -> list[LFConnection]:
    connections: list[LFConnection] = []
    connection_re = re.compile(r"([^\n;{}]+?)\s*->\s*([^\n;{}]+?)\s*;")
    for match in connection_re.finditer(body):
        left_items = _split_csv(match.group(1))
        right_items = _split_csv(match.group(2))
        for left in left_items:
            source_instance, source_port = _parse_endpoint(left)
            for right in right_items:
                target_instance, target_port = _parse_endpoint(right)
                if not source_port or not target_port:
                    continue
                connections.append(
                    LFConnection(
                        source_instance=source_instance,
                        source_port=source_port,
                        target_instance=target_instance,
                        target_port=target_port,
                    )
                )
    return connections


def _populate_state_writes(reactor: LFReactor) -> None:
    state_map = reactor.state_map()
    for reaction in reactor.reactions:
        for state_name, state in state_map.items():
            assignments, updates = _extract_state_writes(state_name, reaction.body)
            for expr in assignments:
                if expr not in state.assignment_expressions:
                    state.assignment_expressions.append(expr)
            for expr in updates:
                if expr not in state.update_expressions:
                    state.update_expressions.append(expr)


def _extract_state_writes(state_name: str, body: str) -> tuple[list[str], list[str]]:
    assignments: list[str] = []
    updates: list[str] = []
    assignment_re = re.compile(
        rf"\b(?:self->)?{re.escape(state_name)}\s*=\s*([^;]+);",
        re.IGNORECASE,
    )
    update_re = re.compile(
        rf"\b(?:self->)?{re.escape(state_name)}\s*([+\-*/%]?=)\s*([^;]+);",
        re.IGNORECASE,
    )
    increment_re = re.compile(
        rf"\b(?:self->)?{re.escape(state_name)}\s*(\+\+|--)",
        re.IGNORECASE,
    )
    for match in assignment_re.finditer(body):
        expr = match.group(1).strip()
        if expr not in assignments:
            assignments.append(expr)
    for match in update_re.finditer(body):
        operator = match.group(1)
        expr = match.group(2).strip()
        if operator == "=":
            continue
        rendered = f"{operator} {expr}"
        if rendered not in updates:
            updates.append(rendered)
    for match in increment_re.finditer(body):
        operator = match.group(1)
        if operator not in updates:
            updates.append(operator)
    return assignments, updates


def _iter_reactor_blocks(original_code: str, stripped_code: str) -> list[tuple[str, str, str, bool]]:
    blocks: list[tuple[str, str, str, bool]] = []
    reactor_re = re.compile(r"\b(main\s+)?reactor(?:\s+([A-Za-z_]\w*))?\s*\{", re.IGNORECASE)
    for match in reactor_re.finditer(stripped_code):
        is_main = bool(match.group(1))
        reactor_name = match.group(2) or ("main" if is_main else "anonymous_reactor")
        open_brace_index = match.end() - 1
        close_brace_index = _find_matching_brace(stripped_code, open_brace_index)
        body_start = open_brace_index + 1
        body_end = close_brace_index
        blocks.append(
            (
                reactor_name,
                original_code[body_start:body_end],
                stripped_code[body_start:body_end],
                is_main,
            )
        )
    return blocks


def _find_matching_brace(text: str, open_brace_index: int) -> int:
    depth = 0
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Unbalanced braces while parsing LF reactor blocks.")


def _strip_c_bodies(text: str) -> str:
    chars = list(text)
    index = 0
    while index < len(chars) - 1:
        if chars[index] == "{" and chars[index + 1] == "=":
            end = text.find("=}", index + 2)
            if end == -1:
                break
            for body_index in range(index, end + 2):
                chars[body_index] = " "
            index = end + 2
            continue
        index += 1
    return "".join(chars)


def _split_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _extract_state_initializer(type_text: str) -> str | None:
    if "=" in type_text:
        return type_text.split("=", 1)[1].strip()
    paren_match = re.search(r"\(([^()]*)\)\s*$", type_text)
    if paren_match:
        return paren_match.group(1).strip()
    return None


def _parse_endpoint(endpoint: str) -> tuple[str, str]:
    cleaned = endpoint.strip()
    if "." not in cleaned:
        return ("self", cleaned)
    instance, port = cleaned.rsplit(".", 1)
    return (instance.strip(), port.strip())


def _normalize_symbol_name(symbol: str) -> str:
    return symbol.strip().split(".")[-1]


def _is_zero_delay_text(text: str) -> bool:
    lowered = " ".join(text.strip().lower().split())
    return lowered in {
        "0",
        "0 ms",
        "0 msec",
        "0 millisecond",
        "0 milliseconds",
        "0 s",
        "0 sec",
        "0 second",
        "0 seconds",
        "0 us",
        "0 usec",
        "0 microsecond",
        "0 microseconds",
        "0 ns",
        "0 nsec",
        "0 nanosecond",
        "0 nanoseconds",
    }


def _looks_like_finite_assignment(expr: str) -> bool:
    stripped = expr.strip().lower()
    if re.fullmatch(r"-?\d+", stripped):
        return True
    if stripped in {"true", "false"}:
        return True
    if re.fullmatch(r"'[^']*'", expr.strip()):
        return True
    return False


def _looks_like_reset_expression(expr: str, initializer: str | None) -> bool:
    stripped = expr.strip().lower()
    if _looks_like_finite_assignment(expr):
        return True
    if initializer and stripped == initializer.strip().lower():
        return True
    return False


def _looks_like_saturating_expression(expr: str, state_name: str) -> bool:
    lowered = expr.lower()
    state_lower = state_name.lower()
    if any(token in lowered for token in ("min(", "max(", "clamp(", "%")):
        return True
    if "?" in expr and ":" in expr and state_lower in lowered:
        return True
    if re.search(rf"{re.escape(state_name)}\s*[<>]=?\s*-?\d+", expr, re.IGNORECASE):
        return True
    return False


def _has_state_based_stabilizing_branch(state_name: str, body: str) -> bool:
    return re.search(
        rf"if\s*\([^)]*{re.escape(state_name)}[^)]*(<=|>=|<|>|==)[^)]*\)\s*\{{[^{{}}]*{re.escape(state_name)}\s*=",
        body,
        re.IGNORECASE | re.DOTALL,
    ) is not None


def _is_boolean_domain(type_text: str) -> bool:
    return type_text.strip().lower().startswith(("bool", "boolean"))


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered
