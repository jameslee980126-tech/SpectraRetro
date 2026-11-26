"""Stateful, budgeted route search with optional evidence-guided decisions.

The controller selects existing search states; it never supplies molecules.
The molecular oracle and hard checks are identical when feedback is masked.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any

from .planner import Planner, Step, canonical_smiles


@dataclass(frozen=True)
class FeedbackConfig:
    max_model_calls: int = 50
    max_states: int = 2000
    max_steps: int = 12
    max_routes: int = 8
    max_controller_calls: int = 12
    max_tool_calls: int = 16
    max_decisions: int = 5000
    max_observation_states: int = 16
    max_feedback_entries: int = 24
    expansion_top_k: int = 40
    use_soft_penalty: bool = False

    def __post_init__(self):
        positive = (self.max_model_calls, self.max_states, self.max_steps,
                    self.max_routes, self.max_decisions, self.max_observation_states,
                    self.max_feedback_entries, self.expansion_top_k)
        if any(type(x) is not int or x < 1 for x in positive):
            raise ValueError("Positive integer search limits are required")
        if any(type(x) is not int or x < 0 for x in
               (self.max_controller_calls, self.max_tool_calls)):
            raise ValueError("Controller/tool limits must be nonnegative integers")
        if type(self.use_soft_penalty) is not bool:
            raise ValueError("use_soft_penalty must be boolean")


class FeedbackResult(dict):
    def to_dict(self):
        return deepcopy(dict(self))


@dataclass
class _SearchState:
    state_id: int
    open_molecules: tuple[str, ...]
    steps: tuple[Step, ...]
    model_rank_cost: float
    soft_penalty: float = 0.0
    parent_id: int | None = None
    last_evidence: dict = field(default_factory=dict)


def _safe(value):
    """Keep tool records JSON serializable without executing their contents."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return str(value)


def _digest(value):
    data = json.dumps(_safe(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf8")).hexdigest()


def _normal_evidence(raw):
    raw = deepcopy(raw) if isinstance(raw, dict) else {}
    reasons = raw.get("reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    if not isinstance(reasons, (list, tuple)):
        reasons = []
    penalty = raw.get("soft_penalty", 0.0)
    if type(penalty) not in (int, float) or not math.isfinite(penalty):
        penalty = 0.0
    # Unknown/unavailable checks stay neutral unless the tool explicitly rejects.
    return {**_safe(raw), "hard_reject": raw.get("hard_reject") is True,
            "reasons": [str(x) for x in reasons],
            "soft_penalty": max(0.0, float(penalty)),
            "evidence": _safe(raw.get("evidence", {}))}


def _bind_core_components(raw_candidate, candidate):
    """Rebind explicit raw-component contracts after canonical component sorting."""
    metadata = candidate["metadata"] = deepcopy(candidate["metadata"])
    contract = metadata.get("core_contract")
    if not isinstance(contract, dict):
        return candidate
    cores = contract.get("cores", [])
    records = [cores] if isinstance(cores, (str, dict)) else cores
    if not isinstance(records, (list, tuple)):
        records = []
    indexed = [core for core in records if isinstance(core, dict) and core.get("component") is not None]
    if not indexed:
        return candidate
    try:
        raw_parts = raw_candidate.get("precursors")
        if isinstance(raw_parts, str):
            raw_parts = raw_parts.split(".")
        if not isinstance(raw_parts, (list, tuple)):
            raise ValueError("unbound_raw_components")
        original = [canonical_smiles(part) for source in raw_parts for part in source.split(".")]
        available = {}
        for index, smiles in enumerate(candidate["precursors"]):
            available.setdefault(smiles, []).append(index)
        mapping = {}
        for index, smiles in enumerate(original):
            if not available.get(smiles):
                raise ValueError("component_multiset_mismatch")
            mapping[index] = available[smiles].pop(0)
        if any(available.values()):
            raise ValueError("component_multiset_mismatch")
        for core in indexed:
            old = core["component"]
            if type(old) is not int or old not in mapping:
                raise ValueError("invalid_raw_core_component")
        remapped = deepcopy(contract)
        remapped_records = deepcopy(records)
        for core in remapped_records:
            if isinstance(core, dict) and core.get("component") is not None:
                core["component"] = mapping[core["component"]]
        remapped["cores"] = remapped_records
        metadata["core_contract"] = remapped
        metadata["core_contract_binding"] = {
            "status": "bound_to_validated_components",
            "raw_component_smiles": original,
            "validated_component_smiles": list(candidate["precursors"]),
            "raw_to_validated_component": {str(k): v for k, v in mapping.items()}}
    except (ValueError, TypeError, AttributeError):
        metadata["unbound_core_contract"] = metadata.pop("core_contract")
        metadata["core_contract_binding"] = {
            "status": "not_assessed", "reason": "raw_component_contract_cannot_be_bound",
            "hard_contract_disabled": True}
    return candidate


class FeedbackPlanner:
    def __init__(self, model, terminal_policy, evidence, controller=None,
                 config=None, feedback_visible=True):
        self.model = model
        self.terminal_policy = terminal_policy
        self.evidence = evidence
        self.controller = controller
        self.config = config or FeedbackConfig()
        if not isinstance(self.config, FeedbackConfig):
            raise TypeError("config must be FeedbackConfig")
        if type(feedback_visible) is not bool:
            raise ValueError("feedback_visible must be boolean")
        self.feedback_visible = feedback_visible
        self._used = False
        self._cache = {}
        self._terminal_cache = {}
        self._states = {}
        self._seen_states = set()
        self._feedback = []
        self._inspected_states = set()
        self._trace = []
        self._errors = []
        self._counts = dict(model_calls=0, controller_calls=0, inspect_tool_calls=0,
                            decision_steps=0, controller_budget_fallbacks=0,
                            step_reviews=0, route_reviews=0, created_states=0,
                            expanded_states=0, cache_hits=0,
                            hard_rejected_candidates=0, cycle_rejected_candidates=0)

    def _terminal(self, molecule):
        if molecule not in self._terminal_cache:
            try:
                self._terminal_cache[molecule] = bool(self.terminal_policy(molecule))
            except Exception as exc:
                self._errors.append("terminal_policy:" + type(exc).__name__)
                self._terminal_cache[molecule] = False
        return self._terminal_cache[molecule]

    def _priority(self, state):
        penalty = state.soft_penalty if self.config.use_soft_penalty else 0.0
        return (state.model_rank_cost + penalty + .1 * len(state.open_molecules),
                state.state_id)

    def _public_evidence(self, record):
        if self.feedback_visible:
            return deepcopy(record)
        # Hard validity facts are shared. Soft diagnostics/values are inaccessible.
        hard_reasons = []
        if record.get("hard_reject"):
            issues = record.get("issues")
            if isinstance(issues, list) and issues:
                hard_reasons = [issue["code"] for issue in issues
                    if isinstance(issue, dict) and issue.get("severity") == "hard"
                    and isinstance(issue.get("code"), str)]
            else:
                hard_codes = {"invalid_product_graph", "invalid_or_missing_precursor_graph",
                    "self_loop", "route_cycle", "retained_core_connectivity_lost",
                    "retained_core_graph_contract_violated"}
                hard_reasons = [reason for reason in record.get("reasons", [])
                                if reason in hard_codes]
        return {"hard_reject": record.get("hard_reject", False),
                "reasons": list(dict.fromkeys(hard_reasons)),
                "feedback_hidden": True}

    def _remember(self, kind, record, **where):
        entry = dict(feedback_id=len(self._feedback) + 1, kind=kind,
                     **where, result=deepcopy(record))
        self._feedback.append(entry)
        self._trace.append(dict(event="evidence_feedback", **deepcopy(entry)))
        return entry["feedback_id"]

    def _budget(self):
        cfg = self.config
        return {**self._counts,
                "remaining_model_calls": max(0, cfg.max_model_calls - self._counts["model_calls"]),
                "remaining_controller_calls": max(0, cfg.max_controller_calls - self._counts["controller_calls"]),
                "remaining_inspect_tool_calls": max(0, cfg.max_tool_calls - self._counts["inspect_tool_calls"]),
                "remaining_states": max(0, cfg.max_states - self._counts["created_states"]),
                "remaining_route_reviews": max(0, cfg.max_routes - self._counts["route_reviews"])}

    def _observation(self, visible_states):
        cfg = self.config
        states = []
        for state in visible_states:
            states.append({"state_id": state.state_id, "parent_id": state.parent_id,
                           "open_molecules": list(state.open_molecules),
                           "is_complete": not state.open_molecules,
                           "next_operation": "expand" if state.open_molecules else "review_route",
                           "step_count": len(state.steps),
                           "model_rank_cost": state.model_rank_cost,
                           "route_so_far": [{"product": s.product, "precursors": list(s.precursors)}
                                            for s in state.steps],
                           "last_step_check": self._public_evidence(state.last_evidence)})
            states[-1]["inspect_available"] = bool(state.last_evidence) and state.state_id not in self._inspected_states
        memory = [{**{k: deepcopy(v) for k, v in entry.items() if k != "result"},
                   "result": self._public_evidence(entry["result"])}
                  for entry in self._feedback[-cfg.max_feedback_entries:]]
        return {"target": self._target, "allowed_actions": ["expand", "inspect"],
                "states": states, "default_state_id": visible_states[0].state_id,
                "feedback_visible": self.feedback_visible,
                "feedback_revision": len(self._feedback), "feedback_memory": memory,
                "budget": self._budget(),
                "instruction": "Select only an existing state_id. Never supply molecules or reactions."}

    def _choose(self):
        visible = sorted(self._states.values(), key=self._priority)[:self.config.max_observation_states]
        default = visible[0]
        if len(self._states) == 1 and len(default.open_molecules) <= 1:
            self._trace.append(dict(event="forced_search_action", state_id=default.state_id,
                reason="single_state_single_frontier" if default.open_molecules else "single_complete_state",
                action="expand", frontier_index=0, controller_call_consumed=False))
            return default, 0
        if self.controller is None or self._counts["controller_calls"] >= self.config.max_controller_calls:
            if self.controller is not None:
                self._counts["controller_budget_fallbacks"] += 1
                self._trace.append(dict(event="search_policy_fallback", reason="controller_call_budget",
                                        state_id=default.state_id, policy="deterministic_model_rank"))
            return default, 0
        observation = self._observation(visible)
        self._counts["controller_calls"] += 1
        try:
            decision = self.controller.decide(deepcopy(observation))
            if not isinstance(decision, dict):
                raise ValueError("controller_response_not_object")
            state_id = decision.get("state_id", default.state_id)
            action = decision.get("action", "expand")
            if type(state_id) is not int or state_id not in {s.state_id for s in visible}:
                raise ValueError("controller_selected_unknown_state")
            if action not in ("expand", "inspect"):
                raise ValueError("controller_selected_unknown_action")
            selected = self._states[state_id]
            frontier = decision.get("frontier_index", 0)
            if type(frontier) is not int or (selected.open_molecules and
                                            not 0 <= frontier < len(selected.open_molecules)):
                raise ValueError("controller_selected_unknown_frontier")
            status = "accepted"
        except Exception as exc:
            selected, frontier, action = default, 0, "expand"
            decision = {}
            status = "fallback:" + type(exc).__name__
            self._errors.append("controller:" + str(exc)[:200])
        feedback_ids = [entry["feedback_id"] for entry in observation["feedback_memory"]]
        trace = dict(event="controller_action", call=self._counts["controller_calls"],
                     action=action, state_id=selected.state_id, frontier_index=frontier,
                     default_state_id=default.state_id,
                     changed_from_default=selected.state_id != default.state_id or frontier != 0,
                     status=status, feedback_visible=self.feedback_visible,
                     feedback_revision=observation["feedback_revision"],
                     observed_feedback_ids=feedback_ids,
                     observation_sha256=_digest(observation), decision=_safe(decision))
        self._trace.append(trace)
        if (action == "inspect" and selected.last_evidence
                and selected.state_id not in self._inspected_states
                and self._counts["inspect_tool_calls"] < self.config.max_tool_calls):
            self._counts["inspect_tool_calls"] += 1
            self._inspected_states.add(selected.state_id)
            self._remember("cached_state_inspection", selected.last_evidence, state_id=selected.state_id)
            return None, 0
        if action == "inspect":
            trace["action_after_budget"] = "expand"
            trace["inspect_bypass_reason"] = ("already_inspected" if selected.state_id in self._inspected_states
                else "no_cached_state_evidence" if not selected.last_evidence else "inspect_tool_budget")
        return selected, frontier

    def _review_step(self, product, candidate, state):
        self._counts["step_reviews"] += 1
        try:
            record = self.evidence.inspect_step(product, deepcopy(candidate),
                         context_steps=[asdict(s) for s in state.steps])
            return _normal_evidence(record)
        except Exception as exc:
            self._errors.append("inspect_step:" + type(exc).__name__)
            return _normal_evidence({"status": "tool_error", "tool_error": type(exc).__name__})

    def _review_route(self, route):
        self._counts["route_reviews"] += 1
        try:
            return _normal_evidence(self.evidence.review_route(deepcopy(route)))
        except Exception as exc:
            self._errors.append("review_route:" + type(exc).__name__)
            return _normal_evidence({"status": "tool_error", "tool_error": type(exc).__name__})

    def _put_state(self, open_molecules, steps, cost, penalty, parent_id, evidence):
        signature = (open_molecules, tuple((s.product, tuple(s.precursors)) for s in steps))
        if signature in self._seen_states:
            return False
        if self._counts["created_states"] >= self.config.max_states:
            return False
        state_id = self._counts["created_states"]
        self._counts["created_states"] += 1
        self._seen_states.add(signature)
        self._states[state_id] = _SearchState(state_id, open_molecules, steps, cost,
                                             penalty, parent_id, deepcopy(evidence))
        self._trace.append(dict(event="state_created", state_id=state_id,
                                parent_id=parent_id, step_count=len(steps),
                                open_molecules=list(open_molecules)))
        return True

    def plan(self, target):
        if self._used:
            raise RuntimeError("A FeedbackPlanner is a single budgeted session; create a new session explicitly")
        self._used = True
        self._target = canonical_smiles(target)
        # The root is always expanded, even if a simple target meets the endpoint.
        self._put_state((self._target,), (), 0.0, 0.0, None, {})
        routes, reviewed_routes, partials, route_hashes = [], [], [], set()
        termination = "search_exhausted"
        while (self._states and self._counts["route_reviews"] < self.config.max_routes
               and self._counts["decision_steps"] < self.config.max_decisions):
            self._counts["decision_steps"] += 1
            state, frontier_index = self._choose()
            if state is None:
                continue
            self._states.pop(state.state_id)
            self._counts["expanded_states"] += 1
            if not state.open_molecules:
                if not state.steps:
                    continue
                route = {"target": self._target, "steps": [asdict(s) for s in state.steps],
                         "starting_materials": sorted({p for s in state.steps for p in s.precursors
                                                       if self._terminal(p)}),
                         "score": state.model_rank_cost, "soft_penalty": state.soft_penalty}
                route_hash = _digest(sorted((s.product, sorted(s.precursors)) for s in state.steps))
                if route_hash in route_hashes:
                    self._trace.append(dict(event="duplicate_route", state_id=state.state_id,
                                            route_hash=route_hash))
                    continue
                route_hashes.add(route_hash)
                record = self._review_route(route)
                route.update(route_hash=route_hash, route_feedback=record)
                feedback_id = self._remember("complete_route_review", record,
                    state_id=state.state_id, route_hash=route_hash,
                    route_summary=[{"product": s.product, "precursors": list(s.precursors)}
                                   for s in state.steps])
                reviewed_routes.append(deepcopy(route))
                if not record["hard_reject"]:
                    routes.append(route)
                self._trace.append(dict(event="route_reviewed", state_id=state.state_id,
                    route_hash=route_hash, accepted=not record["hard_reject"], feedback_id=feedback_id))
                continue
            if len(state.steps) >= self.config.max_steps:
                partials.append(self._partial(state, "step_budget"))
                continue
            product = state.open_molecules[frontier_index]
            if product not in self._cache:
                if self._counts["model_calls"] >= self.config.max_model_calls:
                    partials.append(self._partial(state, "model_query_budget"))
                    continue
                self._counts["model_calls"] += 1
                try:
                    raw = self.model.predict(product, self.config.expansion_top_k)
                    validated = Planner._validate_candidates(product, raw)[:self.config.expansion_top_k]
                    self._cache[product] = [_bind_core_components(raw[row["rank"] - 1], row)
                                            for row in validated]
                except Exception as exc:
                    self._cache[product] = []
                    self._errors.append("predict:" + type(exc).__name__)
                self._trace.append(dict(event="model_query", product=product,
                    logical_query=self._counts["model_calls"], state_id=state.state_id,
                    candidates=len(self._cache[product])))
            else:
                self._counts["cache_hits"] += 1
                self._trace.append(dict(event="model_cache_hit", product=product, state_id=state.state_id))
            candidates = self._cache[product]
            if not candidates:
                partials.append(self._partial(state, "no_candidates_for_open_precursor"))
            for index, candidate in enumerate(candidates):
                # Route-context evidence is recomputed; the molecular candidates stay frozen.
                record = self._review_step(product, candidate, state)
                feedback_id = self._remember("candidate_step_review", record,
                    state_id=state.state_id, product=product, candidate_index=index,
                    precursors=list(candidate["precursors"]))
                if record["hard_reject"]:
                    self._counts["hard_rejected_candidates"] += 1
                    continue
                precursors = candidate["precursors"]
                if Planner._creates_cycle(state.steps, product, precursors):
                    self._counts["cycle_rejected_candidates"] += 1
                    continue
                expanded = {s.product for s in state.steps}
                remaining = tuple(p for p in precursors if not self._terminal(p) and p not in expanded)
                other = state.open_molecules[:frontier_index] + state.open_molecules[frontier_index + 1:]
                new_open = tuple(dict.fromkeys(other + remaining))
                metadata = deepcopy(candidate["metadata"])
                metadata["feedback_step_evidence"] = deepcopy(record)
                metadata["feedback_id"] = feedback_id
                step = Step(product, list(precursors), candidate["score"], candidate["rank"],
                            candidate["source"], metadata)
                created = self._put_state(new_open, state.steps + (step,),
                    state.model_rank_cost + index + 1, state.soft_penalty + record["soft_penalty"],
                    state.state_id, record)
                if not created and self._counts["created_states"] >= self.config.max_states:
                    self._trace.append(dict(event="state_budget", state_id=state.state_id,
                                            candidate_index=index))
                    break
        if self._counts["decision_steps"] >= self.config.max_decisions and self._states:
            termination = "decision_budget"
        elif self._counts["route_reviews"] >= self.config.max_routes:
            termination = "route_review_budget"
        elif self._counts["created_states"] >= self.config.max_states:
            termination = "state_budget"
        elif self._counts["model_calls"] >= self.config.max_model_calls:
            termination = "model_query_budget_or_search_exhausted"
        routes.sort(key=lambda r: (r["score"], r["route_hash"]))
        return FeedbackResult(target=self._target, solved=bool(routes), routes=routes,
            reviewed_routes=reviewed_routes, model_calls=self._counts["model_calls"],
            expanded_products=sorted(self._cache),
            candidate_hit_products=sorted(p for p, rows in self._cache.items() if rows),
            empty_candidate_products=sorted(p for p, rows in self._cache.items() if not rows),
            stock_size=0, errors=list(self._errors), trace=deepcopy(self._trace),
            partial_routes=partials, pending_states=len(self._states), termination=termination,
            feedback_memory=deepcopy(self._feedback), feedback_visible=self.feedback_visible,
            controller_enabled=self.controller is not None, config=asdict(self.config),
            budget=self._budget(),
            protocol="stateful_feedback_planning_v1",
            zero_step_target_closure_allowed=False,
            model_scores_unchanged=True,
            logical_query_accounting="One first product query per arm; shared physical oracle cache does not waive it")

    @staticmethod
    def _partial(state, reason):
        return {"steps": [asdict(s) for s in state.steps], "score": state.model_rank_cost,
                "unresolved_precursors": list(state.open_molecules), "termination": reason,
                "state_id": state.state_id}
