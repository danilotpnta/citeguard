# graph_builder.py

import yaml
import importlib
from typing import Union
from langgraph.graph import StateGraph, END

from app.core.logging import get_logger
from app.utils.graph.build_report import print_graph_summary

logger = get_logger(__name__)


class GraphBuilder:

    def __init__(
        self,
        config: Union[dict, str],
        state_schema,
        modules: list,
    ):
        self.graph_config = self._load_config(config)["graph"]
        self.state_schema = state_schema
        self.modules = modules
        self.registry = self._load_functions()
        self.passthrough_nodes: set[str] = set()  # populated during build()
        self.app = None  # populated after build()

    # ── Config & Registry ─────────────────────────────────────────────────────

    def _load_config(self, config: Union[dict, str]) -> dict:
        if isinstance(config, str):
            with open(config) as f:
                return yaml.safe_load(f)
        return config

    def _load_functions(self) -> dict:
        registry = {}
        for module_path in self.modules:
            module = importlib.import_module(module_path)
            for name in dir(module):
                if name.startswith("_"):
                    continue
                obj = getattr(module, name)
                if callable(obj) and getattr(obj, "__module__", None) == module_path:
                    if name in registry:
                        raise ValueError(
                            f"Duplicate function name '{name}' found in {module_path}"
                        )
                    registry[name] = obj
        return registry

    # ── Shared Helpers ────────────────────────────────────────────────────────

    def _resolve_if_returns(self, if_returns: dict) -> dict:
        return {
            key: END if value == "END" else value for key, value in if_returns.items()
        }

    def _collect_mentioned_nodes(self, gc: dict, subgraph_names: set = None) -> set:
        """
        Collect all node names referenced in a graph config.
        Works identically for parent graphs and subgraphs.

        Args:
            gc: Graph config dict (top-level or subgraph section)
            subgraph_names: Names to exclude (registered as compiled subgraph nodes)
        """
        if subgraph_names is None:
            subgraph_names = set()

        mentioned = set()

        for from_node, to_node in gc.get("edges", {}).items():
            mentioned.add(from_node)
            if to_node != "END":
                mentioned.add(to_node)

        for edge in gc.get("fanout", []):
            mentioned.add(edge["from"])
            for node in edge["to"]:
                mentioned.add(node)

        for edge in gc.get("join", []):
            for node in edge["from"]:
                mentioned.add(node)
            mentioned.add(edge["to"])

        for ce in gc.get("conditional_edges", []):
            mentioned.add(ce["from"])
            for value in ce["if_returns"].values():
                if value != "END":
                    mentioned.add(value)

        mentioned.add(gc["entry_point"])
        mentioned -= subgraph_names

        return mentioned

    def _build_graph(
        self,
        gc: dict,
        label: str = "root",
        print_report: bool = False,
    ) -> StateGraph:
        """
        Build and compile a graph from a config dict.

        This single method handles both the parent graph and any subgraph.
        It supports ALL config features: edges, fanout, barrier joins,
        conditional edges (with optional `then`), passthrough nodes,
        and nested subgraphs (recursive).

        Args:
            gc: Graph config dict
            label: For error/debug messages ("root" or subgraph name)
        """
        builder = StateGraph(self.state_schema)
        edges_added = set()
        barrier_joins_added = set()

        # ── local edge helpers ────────────────────────────────────────────

        def add_edge(from_node: str, to_node) -> None:
            key = (from_node, to_node)
            if key in edges_added:
                logger.warning(
                    f"[{label}] Duplicate edge '{from_node}' -> '{to_node}' ignored"
                )
                return
            builder.add_edge(from_node, to_node)
            edges_added.add(key)

        def add_barrier_join(from_nodes: list, to_node: str) -> None:
            """
            builder.add_edge(["A", "B"], "C") — list syntax.
            Waits for ALL sources, unlike individual calls which fire on ANY.
            """
            key = (tuple(sorted(from_nodes)), to_node)
            if key in barrier_joins_added:
                logger.warning(
                    f"[{label}] Duplicate barrier join {from_nodes} -> '{to_node}' ignored"
                )
                return
            builder.add_edge(from_nodes, to_node)
            barrier_joins_added.add(key)

        # ── 1. Recursively build nested subgraphs ─────────────────────────
        compiled_subgraphs = {}
        for sg_name, sg_config in gc.get("subgraphs", {}).items():
            compiled_subgraphs[sg_name] = self._build_graph(
                sg_config,
                label=sg_name,
                print_report=print_report,
            )

        subgraph_names = set(compiled_subgraphs.keys())

        # ── 2. Collect and register nodes ─────────────────────────────────
        mentioned = self._collect_mentioned_nodes(gc, subgraph_names)

        for name in gc.get("passthrough_nodes", []):
            builder.add_node(name, lambda s: {})
            mentioned.discard(name)
            self.passthrough_nodes.add(name)

        for sg_name, sg_compiled in compiled_subgraphs.items():
            builder.add_node(sg_name, sg_compiled)

        for name in mentioned:
            if name not in self.registry:
                raise ValueError(
                    f"[{label}] Node '{name}' is referenced in YAML but has no "
                    f"matching function.\nAvailable: {sorted(self.registry.keys())}"
                )
            builder.add_node(name, self.registry[name])

        # ── 3. Entry point ────────────────────────────────────────────────
        builder.set_entry_point(gc["entry_point"])

        # ── 4. Regular edges ──────────────────────────────────────────────
        for from_node, to_node in gc.get("edges", {}).items():
            add_edge(from_node, END if to_node == "END" else to_node)

        # ── 5. Fanout edges ───────────────────────────────────────────────
        for edge in gc.get("fanout", []):
            for to_node in edge["to"]:
                add_edge(edge["from"], to_node)

        # ── 6. Barrier joins ─────────────────────────────────────────────
        for edge in gc.get("join", []):
            add_barrier_join(edge["from"], edge["to"])

        # ── 7. Conditional edges ──────────────────────────────────────────
        for ce in gc.get("conditional_edges", []):
            decision_fn_name = ce["decision_fn"]
            if decision_fn_name not in self.registry:
                raise ValueError(
                    f"[{label}] Decision function '{decision_fn_name}' not found."
                    f"\nAvailable: {sorted(self.registry.keys())}"
                )
            kwargs = {}
            if "then" in ce:
                kwargs["then"] = END if ce["then"] == "END" else ce["then"]

            builder.add_conditional_edges(
                ce["from"],
                self.registry[decision_fn_name],
                self._resolve_if_returns(ce["if_returns"]),
                **kwargs,
            )

        # ── Debug output ──────────────────────────────────────────────────
        if print_report:
            print_graph_summary(
                label, gc, compiled_subgraphs, edges_added, barrier_joins_added
            )

        return builder.compile()

    # ── Properties ────────────────────────────────────────────────────────────────

    @property
    def node_registry(self) -> dict:
        """Registry filtered to only real nodes — passthrough lambdas excluded."""
        return {
            k: v for k, v in self.registry.items() if k not in self.passthrough_nodes
        }

    # ── Public ────────────────────────────────────────────────────────────────

    def build(self, print_report: bool = False) -> StateGraph:
        self.app = self._build_graph(
            self.graph_config,
            label="root",
            print_report=print_report,
        )
        return self.app
