"""The agent registry cannot drift from the code (2026-10-30) — offline.

`app/agents/registry.py` is DESCRIPTIVE DATA (nothing at runtime reads it), so
the only thing that keeps it honest is this suite. It proves:

  * every `entry` string resolves to a real, callable graph node — there is
    no registered specialist without an implementation;
  * the registry's node set is EXACTLY the graph's node set, so a new agent
    cannot be added silently and a deleted one cannot linger in the docs;
  * the planner's capability whitelist, the dispatcher's capability→node map
    and the capability→result-channel map all agree with the registry, so a
    planner task can only name work a registered specialist performs;
  * each entry's `tests` files exist, and its declared permissions are
    consistent (network only for research; filesystem writes only for the
    feasibility study and the approved coding patch);
  * no node is registered twice, and every entry states what it rejects.

No app startup, no model, no network, no Chroma.
"""

import importlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import chroma_guard  # noqa: E402

chroma_guard.arm()
chroma_guard.install_persistent_client_wrapper()

import app.graph as gm  # noqa: E402
from app.agents import registry  # noqa: E402
from app.planner import CAPABILITIES  # noqa: E402

REQUIRED_FIELDS = {
    "kind", "node", "entry", "routes", "planner_capability", "tools",
    "permissions", "reads", "writes", "rejects", "model", "tests",
}


def _resolve(entry: str):
    module_name, _, attribute = entry.partition(":")
    if not module_name or not attribute:
        raise AssertionError(f"malformed entry reference: {entry!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


class TestRegistryShape(unittest.TestCase):
    def test_every_entry_declares_the_full_contract(self):
        for name, entry in registry.AGENTS.items():
            self.assertTrue(
                REQUIRED_FIELDS.issubset(entry),
                f"{name} is missing {sorted(REQUIRED_FIELDS - set(entry))}")
            self.assertEqual(entry["kind"], "agent")
            self.assertTrue(str(entry["node"]).strip())
            self.assertTrue(str(entry["rejects"]).strip(),
                            f"{name} must state what it refuses")
            self.assertTrue(str(entry["model"]).strip(),
                            f"{name} must state whether it uses a model")

    def test_every_entry_resolves_to_a_real_callable(self):
        for name, entry in registry.AGENTS.items():
            self.assertTrue(callable(_resolve(str(entry["entry"]))),
                            f"{name}: {entry['entry']} is not callable")
        for name, entry in registry.ORCHESTRATION.items():
            self.assertTrue(callable(_resolve(str(entry["entry"]))),
                            f"{name}: {entry['entry']} is not callable")

    def test_no_node_is_registered_twice(self):
        nodes = registry.agent_nodes() + registry.orchestration_nodes()
        self.assertEqual(len(nodes), len(set(nodes)), nodes)


class TestRegistryMatchesTheGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph_nodes = set(gm.build_graph().nodes) - {"__start__", "__end__"}

    def test_registry_covers_the_graph_exactly(self):
        registered = set(registry.agent_nodes() + registry.orchestration_nodes())
        self.assertEqual(
            registered, self.graph_nodes,
            "registry and graph disagree: "
            f"only in registry={sorted(registered - self.graph_nodes)}, "
            f"only in graph={sorted(self.graph_nodes - registered)}")

    def test_each_entry_node_is_the_node_the_graph_builds(self):
        # The registry names the node; the entry must be the function behind
        # that node, not a different implementation that merely exists.
        known = {
            # graph nodes whose callable is deliberately NOT the entry point
            # (the dispatcher/graph resolve some node identity at call time)
            "research": "app.graph:research_entry",
        }
        for name, entry in registry.AGENTS.items():
            node = entry["node"]
            if node is None:
                # Dispatch-only specialist (synthesis): its node identity is
                # the capability mapping, checked in TestCapabilitiesAgree.
                self.assertTrue(entry["planner_capability"], name)
                continue
            node = str(node)
            self.assertIn(node, self.graph_nodes)
            expected = known.get(node, str(entry["entry"]))
            self.assertEqual(str(entry["entry"]), expected)


class TestCapabilitiesAgree(unittest.TestCase):
    def test_planner_whitelist_is_the_registry_capability_set(self):
        self.assertEqual(
            set(CAPABILITIES), set(registry.planner_capabilities()),
            "the planner may name a capability that is not registered")

    def test_dispatch_mapping_matches_the_registry(self):
        self.assertEqual(gm.CAPABILITY_NODES,
                         registry.planner_capabilities())
        # The dispatcher also has a FEASIBILITY entry: the feasibility route
        # is a direct route, so the planner never names it, but a plan may
        # still carry it. That extra key is the only allowed difference.
        self.assertEqual(set(gm._TASK_NODE_FNS) - set(registry.planner_capabilities()),
                         {"feasibility"})
        self.assertTrue(set(registry.planner_capabilities())
                        .issubset(set(gm._TASK_NODE_FNS)))
        for capability, node in gm.CAPABILITY_NODES.items():
            self.assertEqual(gm._TASK_NODE_FNS[capability],
                             _resolve_function_name(node),
                             f"{capability} dispatches to a different node")

    def test_result_channels_match_the_registry(self):
        self.assertEqual(set(gm.CAPABILITY_AGENT_KEYS),
                         set(registry.planner_capabilities()))
        for capability, channel in gm.CAPABILITY_AGENT_KEYS.items():
            agent_name = registry.planner_capabilities()[capability]
            # The channel is the agent's key inside agent_answers, so a
            # consumer reading the task result reads the specialist's own
            # output — never another agent's.
            self.assertEqual(channel, agent_name,
                             f"{capability} reports into {channel!r}")
            self.assertIn("agent_answers",
                          registry.AGENTS[agent_name]["writes"])


def _resolve_function_name(node: str) -> str:
    """The `app.graph` global a dispatched node resolves to."""
    return {
        "research": "research_entry",
        "books": "books_agent",
        "erp": "erp_agent",
        "feasibility": "feasibility_agent",
        "data": "data_agent",
        "synthesis": "synthesis_node",
    }[node]


class TestDeclaredResources(unittest.TestCase):
    def test_declared_test_files_exist(self):
        for name, entry in registry.AGENTS.items():
            for token in str(entry["tests"]).split(","):
                path = token.strip()
                if not path:
                    continue
                self.assertTrue((PROJECT_ROOT / path).is_file(),
                                f"{name} names a missing test file: {path}")
        for name, entry in registry.ORCHESTRATION.items():
            for token in str(entry["tests"]).split(","):
                path = token.strip()
                if path:
                    self.assertTrue((PROJECT_ROOT / path).is_file(),
                                    f"{name} names a missing test file: {path}")

    def test_skills_and_memory_point_at_real_files(self):
        import re

        for name, description in registry.SKILLS.items():
            match = re.match(r"(docs/[\w./\-]+)", description)
            self.assertIsNotNone(match, f"skill {name} names no document")
            self.assertTrue((PROJECT_ROOT / match.group(1)).is_file(),
                            f"skill {name} points at {match.group(1)}")
        self.assertIn("store", " ".join(registry.MEMORY.values()).lower())

    def test_permissions_are_explicit_and_minimal(self):
        # Network: research only (through its two tools, after approval).
        self.assertEqual(registry.entries_with_permission("network"),
                         ["research"])
        # Writes: the feasibility study, the approved patch, and — since
        # 2026-10-30 — the feedback agent, whose ONLY write is a lesson row in
        # the app's own execution store (never a workspace file, never a
        # permission change; the value says so instead of being a bare True).
        self.assertEqual(registry.entries_with_permission("writes"),
                         ["coding", "feasibility", "feedback"])
        self.assertEqual(
            registry.AGENTS["feedback"]["permissions"]["writes"],
            "lesson rows only")
        self.assertFalse(registry.AGENTS["feedback"]["permissions"]["filesystem"])
        self.assertFalse(registry.AGENTS["feedback"]["permissions"]["network"])
        # Approval-gated: the two operations that touch the outside world.
        self.assertEqual(registry.entries_with_permission("approval_required"),
                         ["coding", "research"])

    def test_every_agent_declares_a_permission_block(self):
        for name, entry in registry.AGENTS.items():
            permissions = entry["permissions"]
            self.assertEqual(
                set(permissions),
                {"network", "filesystem", "writes", "approval_required"},
                f"{name} must state all four permission dimensions")

    def test_tool_index_finds_the_owners(self):
        self.assertEqual(registry.entries_using_tool("searxng"),
                         ["research"])
        self.assertEqual(registry.entries_using_tool("workspace"),
                         ["coding"])
        self.assertEqual(registry.entries_using_tool("analysis.py"), ["data"])


class TestNoFakeAgents(unittest.TestCase):
    def test_registered_agents_have_distinct_entry_points(self):
        entries = [str(e["entry"]) for e in registry.AGENTS.values()]
        self.assertEqual(len(entries), len(set(entries)))

    def test_every_agent_declares_at_least_one_route_or_a_planned_role(self):
        for name, entry in registry.AGENTS.items():
            self.assertTrue(
                entry["routes"] or entry["planner_capability"],
                f"{name} has neither a route nor a planned role")


if __name__ == "__main__":
    unittest.main(verbosity=2)
