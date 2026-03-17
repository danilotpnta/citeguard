def print_graph_summary(
    label,
    gc,
    compiled_subgraphs,
    edges_added,
    barrier_joins_added,
) -> None:
    """Print a summary of the graph structure. Works for root and subgraphs."""
    indent = "" if label == "root" else "  │ "

    if label == "root":
        print("=" * 60)
    else:
        print(f"\n  ┌─ Subgraph: {label}")

    if compiled_subgraphs:
        print(f"{indent}SUBGRAPHS:")
        for sg_name in compiled_subgraphs:
            print(f"{indent}  {sg_name}")

    if edges_added:
        print(f"{indent}REGULAR EDGES:")
        for edge in sorted(edges_added):
            print(f"{indent}  {edge[0]} → {edge[1]}")

    if barrier_joins_added:
        print(f"{indent}BARRIER JOINS:")
        for join_key in sorted(barrier_joins_added):
            from_nodes, to_node = join_key
            print(f"{indent}  {list(from_nodes)} → {to_node}")

    conditional_edges = gc.get("conditional_edges", [])
    if conditional_edges:
        print(f"{indent}CONDITIONAL EDGES:")
        for ce in conditional_edges:
            then_str = f" (then → {ce['then']})" if "then" in ce else ""
            print(f"{indent}  {ce['from']} → {ce['if_returns']}{then_str}")

    if label == "root":
        print("=" * 60)
    else:
        print(f"  └─ /{label}")
