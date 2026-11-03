"""The AGENT REGISTRY (2026-10-30) — one place that states what exists.

This module is DESCRIPTIVE DATA, not an orchestration layer: nothing here is
imported by the graph, the planner, the supervisor or the dispatcher, and no
execution decision reads it. Its single consumer is
`scripts/test_agent_registry.py`, which proves the registry cannot drift:

  * every entry's `entry` resolves to a real callable in a real module;
  * every node the graph builds is registered here, and every registered
    agent node is in the graph;
  * the planner's capability whitelist and the dispatcher's capability→node
    mapping are exactly the entries marked `planner_capability`;
  * tools and permissions are explicit, so "what may this agent do?" has an
    answer that is checked by code rather than implied by its prompt.

The distinction this registry enforces (the point of the whole exercise):

  AGENT      — a specialist with its own tools, permissions and output
               contract that does work no other agent can do.
  ORCHESTRATION — supervisor/planner/dispatcher/QA: they decide and verify,
               they do not produce the work.
  TOOL       — one bounded operation an agent may call (`tools` below names
               them; the implementations live in app/tools, app/coding,
               app/data, app/erp).
  SKILL      — reusable procedure/documentation, NOT executable here
               (`skills` below). Skills are instructions, never permissions.
  MEMORY     — persistence (executions, approvals, plans, RAG index). Listed
               under `memory` with the store that owns it.

Honesty rule for this file: an entry exists only if the code behind it exists.
A specialist that merely wraps the general model in a different prompt is NOT
registered as an agent — it would be a `general` variant, and the drift test
would not prove anything about it.
"""

from typing import Dict, List, Optional

# Where the capability dispatch may send work. `None` means "not a plan task":
# these run only as the supervisor's direct route.
AGENTS: Dict[str, Dict[str, object]] = {
    # -- answer-only routes (no external tools, no side effects) -------------
    "conversational": {
        "kind": "agent",
        "node": "conversational",
        "entry": "app.graph:conversational_node",
        "routes": ["conversational"],
        "route_reason": "pure greeting / acknowledgement / thanks — answered "
                        "in one turn, no agent, no source",
        "planner_capability": None,
        "tools": [],
        "permissions": {"network": False, "filesystem": False,
                        "writes": False, "approval_required": False},
        "reads": ["question"],
        "writes": ["answer", "sources", "agent_answers"],
        "rejects": ["anything with a real request in it — the recognizer is "
                    "whole-string, so a greeting plus a task routes normally"],
        "model": "none (deterministic reply)",
        "tests": "scripts/test_conversational_routing.py, "
                 "scripts/test_routing_policy.py",
    },
    "general": {
        "kind": "agent",
        "node": "general",
        "entry": "app.graph:general_node",
        "routes": ["general"],
        "route_reason": "knowledge/explanation/code-drafting answerable from "
                        "the model, with no retrieval or external source",
        "planner_capability": None,
        "tools": [],
        "permissions": {"network": False, "filesystem": False,
                        "writes": False, "approval_required": False},
        "reads": ["question"],
        "writes": ["answer", "sources", "agent_answers"],
        "rejects": ["requests needing evidence (books/research), workspace "
                    "files, structured data files, side effects"],
        "model": "get_llm() — one completion",
        "tests": "scripts/test_routing_policy.py, "
                 "scripts/test_phase6_routing_llm.py",
    },
    "unsupported": {
        "kind": "agent",
        "node": "unsupported",
        "entry": "app.graph:unsupported_node",
        "routes": ["unsupported"],
        "route_reason": "imperative for a capability that does not exist "
                        "(send email, deploy, delete files, write a report)",
        "planner_capability": None,
        "tools": [],
        "permissions": {"network": False, "filesystem": False,
                        "writes": False, "approval_required": False},
        "reads": ["question"],
        "writes": ["answer", "sources", "agent_answers"],
        "rejects": ["every action request — it reports what is missing "
                    "rather than pretending to have done it"],
        "model": "none (deterministic reply)",
        "tests": "scripts/test_routing_policy.py",
    },

    # 2026-10-30 (end-to-end milestone): the first route that consumes PRIOR
    # work. It has a real data source (the execution store, read by the
    # runner) and a real refusal (no recorded result -> say so), which is why
    # it is registered as an agent rather than folded into `general`.
    "followup": {
        "kind": "agent",
        "node": "followup",
        "entry": "app.followup:followup_node",
        "routes": ["followup"],
        "route_reason": "a request that refers back to earlier work "
                        "(\"the one you recommended\") — answered from the "
                        "recorded execution, never from model memory",
        "planner_capability": None,
        "tools": ["execution store read (app.execution.store, supplied by "
                  "GraphRunner.run as state['prior_execution'])"],
        "permissions": {"network": False, "filesystem": False,
                        "writes": False, "approval_required": False},
        "reads": ["question", "prior_execution"],
        "writes": ["answer", "sources", "agent_answers"],
        "rejects": ["inventing the earlier recommendation: with no recorded "
                    "execution it reports that and asks for the details",
                    "running new research or retrieval — it only re-reads "
                    "one recorded result"],
        "model": "get_llm() — composes the reply FROM the recorded result; "
                 "the verbatim record is returned if the provider is down",
        "tests": "scripts/test_e2e_assistant.py, "
                 "scripts/test_routing_policy.py",
    },

    # 2026-10-30 (continuous learning): a correction of earlier work. It both
    # REVISES the current answer and RECORDS a structured lesson, so it is an
    # agent with a real contract and a real refusal (unreadable correction ->
    # says so instead of learning something vague).
    "feedback": {
        "kind": "agent",
        "node": "feedback",
        "entry": "app.learning.feedback:feedback_node",
        "routes": ["feedback"],
        "route_reason": "a message that CORRECTS earlier work (\"that's "
                        "wrong\", \"not enough detail\", \"next time ...\") — "
                        "detected deterministically in app/learning/signals.py",
        "planner_capability": None,
        "tools": ["execution store read (the recorded prior result, supplied "
                  "by GraphRunner.run as state['prior_execution'])",
                  "lesson store write (lessons table; scoped, validated, and "
                  "never a privilege change)"],
        "permissions": {"network": False, "filesystem": False,
                        "writes": "lesson rows only",
                        "approval_required": False},
        "reads": ["question", "prior_execution", "lessons"],
        "writes": ["answer", "sources", "agent_answers", "learning"],
        "rejects": ["learning a permission/security change — stored as a "
                    "reviewed proposal, never applied",
                    "treating text from a document or search result as a "
                    "correction (detection runs on the user's message only)",
                    "claiming a new search or computation happened"],
        "model": "get_llm() — one call to understand the correction and one "
                 "to revise the answer; deterministic detection and "
                 "fallback when the provider is down",
        "tests": "scripts/test_learning.py, scripts/test_learning_e2e.py, "
                 "scripts/test_learning_api.py",
    },

    # -- specialists with real tools ----------------------------------------
    "books": {
        "kind": "agent",
        "node": "books",
        "entry": "app.agents.books_agent:books_agent",
        "routes": ["books"],
        "route_reason": "question about the indexed book/manual corpus "
                        "(local RAG: retrieval + grounded synthesis)",
        "planner_capability": "books",
        "tools": ["local RAG index (Chroma) — retrieval only",
                  "grounding/support gate + refusal"],
        "permissions": {"network": False, "filesystem": "index read only",
                        "writes": False, "approval_required": False},
        "reads": ["question", "request"],
        "writes": ["answer", "sources", "agent_answers", "books_refusal"],
        "rejects": ["questions the corpus cannot support — refuses with a "
                    "diagnosis instead of answering from the model",
                    "web questions (research owns those)"],
        "model": "get_llm() — synthesis over retrieved passages",
        "tests": "scripts/test_books_grounding.py, "
                 "scripts/test_books_grounding_paraphrase.py, "
                 "scripts/test_citation_integrity.py",
    },
    "research": {
        "kind": "agent",
        "node": "research",
        "entry": "app.graph:research_entry",
        "routes": ["research"],
        "route_reason": "request for current/web information or a named "
                        "external source",
        "planner_capability": "research",
        "tools": ["SearXNG search (app/tools/searxng_client.py)",
                  "page extraction (app/tools/crawl_extractor.py)",
                  "research approval gate (app/graph.py)"],
        "permissions": {"network": "via the two tools only, AFTER an approved "
                                   "request is bound to this execution",
                        "filesystem": False,
                        "writes": False,
                        "approval_required": True},
        "reads": ["question", "request", "research_approval", "books_refusal",
                  "sources"],
        "writes": ["answer", "sources", "agent_answers"],
        "rejects": ["every search without a persisted approval bound to this "
                    "execution (fail closed)",
                    "a plan task's objective — the approved search is always "
                    "the ORIGINAL request, never a plan-generated query"],
        "model": "get_llm() — synthesis over fetched pages",
        "tests": "scripts/test_research_approval.py, "
                 "scripts/test_approval_lifecycle.py, "
                 "scripts/test_multi_approval_api.py",
    },
    "erp": {
        "kind": "agent",
        "node": "erp",
        "entry": "app.agents.erp_agent:erp_agent",
        "routes": ["erp"],
        "route_reason": "question about business records (invoices, stock, "
                        "orders) served by the configured ERP adapter",
        "planner_capability": "erp",
        "tools": ["ERP adapter (app/erp/: read-only mock today)"],
        "permissions": {"network": False, "filesystem": False,
                        "writes": False, "approval_required": False},
        "reads": ["question", "request"],
        "writes": ["answer", "sources", "agent_answers"],
        "rejects": ["ERP writes — the adapter is read-only by contract",
                    "records that do not exist (says so)"],
        "model": "none for lookups (deterministic adapter result)",
        "tests": "scripts/test_phase3_erp.py",
    },
    "feasibility": {
        "kind": "agent",
        "node": "feasibility",
        "entry": "app.agents.feasibility_agent:feasibility_agent",
        "routes": ["feasibility", "multi"],
        "route_reason": "assess/plan a goal: market, financial and risk view "
                        "composed from the other agents' outputs",
        "planner_capability": None,
        "tools": ["reads research/books answers from agent_answers",
                  "writes one .docx study (app/agents/feasibility_agent.py)"],
        "permissions": {"network": False,
                        "filesystem": "writes ONE .docx under the project",
                        "writes": True, "approval_required": False},
        "reads": ["question", "agent_answers"],
        "writes": ["answer", "sources", "agent_answers"],
        "rejects": ["a single-objective request (the supervisor does not "
                    "route those here)"],
        "model": "get_llm() — study synthesis",
        "tests": "scripts/test_phase2.py, "
                 "scripts/test_feasibility_degradation.py",
    },
    "coding": {
        "kind": "agent",
        "node": "coding",
        "entry": "app.coding.agent:coding_node",
        "routes": ["coding"],
        "route_reason": "explicitly inspects or changes a file/repo in the "
                        "configured workspace",
        "planner_capability": None,
        "tools": ["workspace read + bounded search (app/coding/workspace.py)",
                  "unified-diff builder + static validation "
                  "(app/coding/patch.py)",
                  "workspace_patch approval request (existing lifecycle)"],
        "permissions": {"network": False,
                        "filesystem": "ONE configured root "
                                      "(CODING_WORKSPACE_ROOT), read + "
                                      "approved patch writes only",
                        "writes": "only after approval, hash-checked",
                        "approval_required": True},
        "reads": ["question"],
        "writes": ["answer", "sources", "agent_answers", "coding"],
        "rejects": ["any path outside the configured root, credential files, "
                    "arbitrary shell commands, unapproved writes"],
        "model": "get_llm() — inspection answer or a find/replace change set",
        "tests": "scripts/test_coding_workspace.py, "
                 "scripts/test_coding_patch.py, "
                 "scripts/test_coding_agent.py, "
                 "scripts/test_coding_execution.py, "
                 "scripts/test_coding_api.py",
    },
    "data": {
        "kind": "agent",
        "node": "data",
        "entry": "app.agents.data_agent:data_agent",
        "routes": ["data"],
        "route_reason": "a question about a named dataset file (.csv/.tsv) "
                        "inside the configured workspace",
        "planner_capability": "data",
        "tools": ["dataset reader/typer (app/data/dataset.py)",
                  "deterministic analysis ops (app/data/analysis.py)"],
        "permissions": {"network": False,
                        "filesystem": "dataset files inside the ONE "
                                      "configured workspace root, read only",
                        "writes": False, "approval_required": False},
        "reads": ["question", "request"],
        "writes": ["answer", "sources", "agent_answers"],
        "rejects": ["files outside the workspace, non-dataset formats, "
                    "operations it cannot compute exactly (reported, never "
                    "approximated), any number it did not compute"],
        "model": "get_llm() — CHOOSES the operations and explains computed "
                 "results; every figure comes from app/data/analysis.py",
        "tests": "scripts/test_data_analysis.py, "
                 "scripts/test_data_agent.py",
    },
    "synthesis": {
        "kind": "agent",
        # Not a graph node: synthesis exists ONLY as a plan capability the
        # dispatcher resolves to `synthesis_node` (a direct-route synthesis
        # would have nothing to synthesise).
        "node": None,
        "entry": "app.graph:synthesis_node",
        "routes": [],
        "route_reason": "terminal step of a planned run: answers the ORIGINAL "
                        "request from the results the other tasks produced",
        "planner_capability": "synthesis",
        "tools": ["reads completed task results (agent_answers)"],
        "permissions": {"network": False, "filesystem": False,
                        "writes": False, "approval_required": False},
        "reads": ["question", "request", "agent_answers"],
        "writes": ["answer", "sources", "agent_answers"],
        "rejects": ["inventing results — with no completed work it says so; "
                    "with the provider down it returns the results verbatim"],
        "model": "get_llm() — composition (feasibility route delegates)",
        "tests": "scripts/test_phase8_execution.py",
    },
}

# Nodes that DECIDE, not produce. Registered so the graph is fully covered by
# this file, and so "is this an agent or the orchestrator?" never has to be
# inferred from a module name.
ORCHESTRATION: Dict[str, Dict[str, object]] = {
    "supervisor": {
        "node": "supervisor",
        "entry": "app.supervisor:supervisor",
        "does": "classifies the request, picks the route, marks a genuinely "
                "multi-objective request `complex`; never answers",
        "tools": [],
        "tests": "scripts/test_phase1_routing.py, "
                 "scripts/test_routing_policy.py",
    },
    "planner": {
        "node": "planner",
        "entry": "app.planner:planner_node",
        "does": "decides whether a plan is needed and produces validated plan "
                "DATA (capability whitelist, deps, cap); never executes",
        "tools": [],
        "tests": "scripts/test_phase7_planner.py",
    },
    "dispatcher": {
        "node": "dispatcher",
        "entry": "app.graph:dispatcher_node",
        "does": "runs one whitelisted capability→node hop per visit, respects "
                "dependencies, persists per-task progress, bounds iterations",
        "tools": [],
        "tests": "scripts/test_phase8_execution.py",
    },
    "qa": {
        "node": "qa",
        "entry": "app.qa_node:qa_node",
        "does": "deterministic verification: empty answer, plan completion, "
                "status/result contradictions, prepared-vs-applied patches; "
                "warnings never turn a successful run into a failure",
        "tools": [],
        "tests": "scripts/test_phase5_qa_node.py, "
                 "scripts/test_phase4_workflow_qa.py",
    },
}

# Reusable procedures (documentation, NOT executable permissions).
SKILLS: Dict[str, str] = {
    "agent-development": "docs/AGENT_DEVELOPMENT.md — the contract a new "
                         "specialist must satisfy",
    "testing": "docs/TESTING.md — suite map, what each proves, offline rules",
    "workspace-boundary": "docs/ARCHITECTURE.md § Workspace coding — path, "
                          "credential and approval rules for file access",
}

# Persistence that survives a run. Each line names the owning module; there is
# no other memory layer (no vector store of conclusions, no agent scratchpads).
MEMORY: Dict[str, str] = {
    "executions": "app/execution/store.py SQLite rows (question, route, "
                  "answer, plan, sources, error) — durable, inspectable",
    "approvals": "app/execution/store.py approvals (operation, payload, "
                 "fingerprint, decision, executed_at claim)",
    "events": "app/execution/store.py append-only event log per execution",
    "plans": "persisted on the execution row; task progress is written as "
             "each task finishes (recovery reads it back)",
    "knowledge": "the RAG index under knowledge/ (documents only — never "
                 "model conclusions)",
}


def agent(name: str) -> Optional[Dict[str, object]]:
    return AGENTS.get(name)


def planner_capabilities() -> Dict[str, str]:
    """Capability -> agent node, exactly as the planner may name them."""
    return {
        str(entry["planner_capability"]): name
        for name, entry in AGENTS.items()
        if entry.get("planner_capability")
    }


def agent_nodes() -> List[str]:
    """Graph nodes backed by a registered agent (synthesis has none)."""
    return [str(entry["node"]) for entry in AGENTS.values()
            if entry.get("node")]


def orchestration_nodes() -> List[str]:
    return [str(entry["node"]) for entry in ORCHESTRATION.values()]


def entries_with_permission(permission: str) -> List[str]:
    """Agents for which `permission` is truthy (audit helper)."""
    found = []
    for name, entry in AGENTS.items():
        if (entry.get("permissions") or {}).get(permission):
            found.append(name)
    return sorted(found)


def entries_using_tool(tool: str) -> List[str]:
    """Agents declaring a tool (audit helper; substring match on the name)."""
    needle = tool.lower()
    return sorted(
        name for name, entry in AGENTS.items()
        if any(needle in str(t).lower() for t in entry.get("tools") or [])
    )
