from app.graph.object.graph_builder import GraphBuilder
from app.utils.graph.mermaid_renderer import render_langgraph_png
from app.graph.state import WorkflowState
from app.core.config import config

builder = GraphBuilder(
    config=config.pipeline.citeguard.graph_structure,
    state_schema=WorkflowState,
    modules=[
        "app.graph.nodes.input_nodes",
        "app.graph.nodes.extraction_nodes",
        "app.graph.nodes.ai_nodes",
        "app.graph.routes",
    ],
)
citeguard_graph = builder.build()


def save_pipeline_graph(filename="pipeline.png"):
    parent_dir = os.path.dirname(config.pipeline.citeguard.rendered_graph)
    img_path = os.path.join(parent_dir, filename)
    render_langgraph_png(
        app=builder.app,
        registry=builder.registry,
        path=img_path,
    )


if __name__ == "__main__":
    import argparse
    import os

    version = config.pipeline.citeguard.version
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        default=f"pipeline_{version}.png",
    )
    args = parser.parse_args()
    save_pipeline_graph(args.filename)
