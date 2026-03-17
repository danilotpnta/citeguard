import re
import asyncio
from typing import Literal
from mermaid_cli import render_mermaid


# MERMAID DARK THEME COLORS
THEME_BG = "#24292F"  # ------------- Canvas background (entire image)
THEME_NODE_DEFAULT = "#1e1e2e"  # --- Default node background color
THEME_NODE_LAST = "#166F2C"  # ------ END node background (green)
THEME_PRIMARY = "#bd93f9"  # -------- Fallback node fill (themeVariables)
THEME_TEXT = "#ffffff"  # ----------- Text inside default nodes
THEME_TEXT_ALT = "#f8f8f2"  # ------- Text in themeVariables (edge labels, etc.)
THEME_STROKE = "#6c7086"  # --------- Default node border/outline
THEME_STROKE_ALT = "#6272a4"  # ----- END node border & fallback borders
THEME_LINE = "#8be9fd"  # ----------- Arrow/edge line color
THEME_SECONDARY = "#44475a"  # ------ Edge label background
THEME_FILE_PATH = "#ca5e9b"  # ------ File path text color (<small> tags)

# Subgraph-specific colors
THEME_SUBGRAPH_BG = "#2d333b"  # ---- Subgraph container background
THEME_SUBGRAPH_STROKE = "#6c7086"  # - Subgraph container border
THEME_SUBGRAPH_TEXT = "#FFAB70"  # -- Subgraph title text


def inject_file_labels(
    mermaid: str, node_file_map: dict, file_color: str = THEME_FILE_PATH
) -> str:
    """
    Finds node definitions in the Mermaid string and appends the file path
    below the node name as a smaller label with custom color.
    """
    for node_name, file_path in node_file_map.items():
        annotation = (
            f"{node_name}<br/>"
            f'<small><font color="{file_color}">{file_path}</font></small>'
        )
        pattern = (
            rf'([\(\["]+(?:<p>)?)'  # opener: one or more of ( [ " then optional <p>
            rf"({re.escape(node_name)})"  # exact node name as display label
            rf'((?:</p>)?[\)\]"]+)'  # closer: optional </p> then one or more of ) ] "
        )
        mermaid = re.sub(pattern, rf"\g<1>{annotation}\g<3>", mermaid)

    return mermaid


def inject_subgraph_styles(mermaid: str, theme: str = "dark") -> str:
    """
    Find all subgraph names in the Mermaid string and inject style directives
    to match the dark theme. Mermaid's default subgraph styling is light/white
    which clashes with dark backgrounds.

    Injects lines like:
        style content_subgraph fill:#2d333b,stroke:#6c7086,color:#c9d1d9
    """
    if theme != "dark":
        return mermaid

    # Find all subgraph names: "subgraph <name>" lines
    subgraph_names = re.findall(r"^\s*subgraph\s+(\S+)", mermaid, re.MULTILINE)

    if not subgraph_names:
        return mermaid

    # Build style lines for each subgraph
    style_lines = []
    for sg_name in subgraph_names:
        style_lines.append(
            f"    style {sg_name} fill:{THEME_SUBGRAPH_BG},"
            f"stroke:{THEME_SUBGRAPH_STROKE},"
            f"stroke-width:1.5px,"
            f"color:{THEME_SUBGRAPH_TEXT}"
        )

    # Insert style directives before the last line (which is usually just ";")
    # or at the end of the graph definition
    style_block = "\n".join(style_lines)

    # Find the classDef lines (which are near the end) and insert before them
    classdef_match = re.search(r"(\n\s*classDef\s)", mermaid)
    if classdef_match:
        insert_pos = classdef_match.start()
        mermaid = mermaid[:insert_pos] + "\n" + style_block + mermaid[insert_pos:]
    else:
        # Fallback: append before the final line
        mermaid = mermaid.rstrip() + "\n" + style_block + "\n"

    return mermaid


def apply_dark_theme(mermaid: str) -> str:
    """Replace LangGraph's light theme classDefs with dark ones."""
    mermaid = mermaid.replace(
        "classDef default fill:#f2f0ff,line-height:1.2",
        f"classDef default fill:{THEME_NODE_DEFAULT},color:{THEME_TEXT},"
        f"stroke:{THEME_STROKE},line-height:1.2",
    )
    mermaid = mermaid.replace(
        "classDef last fill:#bfb6fc",
        f"classDef last fill:{THEME_NODE_LAST},color:{THEME_TEXT},"
        f"stroke:{THEME_STROKE_ALT},stroke-width:2px",
    )

    # Style subgraph containers
    mermaid = inject_subgraph_styles(mermaid, theme="dark")

    return mermaid


async def _render_mermaid_png(
    mermaid: str, theme: Literal["dark", "light"] = "dark"
) -> bytes:
    """Async render mermaid to PNG bytes."""

    if theme == "dark":
        render_config = {
            "background_color": THEME_BG,
            "mermaid_config": {
                "theme": "base",
                "themeVariables": {
                    "primaryColor": THEME_PRIMARY,
                    "primaryTextColor": THEME_TEXT_ALT,
                    "primaryBorderColor": THEME_STROKE_ALT,
                    "lineColor": THEME_LINE,
                    "secondaryColor": THEME_SECONDARY,
                    "tertiaryColor": THEME_STROKE_ALT,
                    "nodeTextColor": THEME_TEXT_ALT,
                    "edgeLabelBackground": THEME_SECONDARY,
                    "textColor": THEME_TEXT_ALT,
                },
            },
        }
    else:
        render_config = {
            "mermaid_config": {"theme": "default"},
        }

    _, _, png_data = await render_mermaid(
        mermaid,
        output_format="png",
        **render_config,
    )
    return png_data


def render_langgraph_png(
    app,
    registry: dict,
    path: str,
    theme: Literal["dark", "light"] = "dark",
    xray: bool = True,
    show_file_paths: bool = True,
    save_mmd: bool = False,
    debug: bool = False,
) -> None:
    """
    Render a LangGraph app to PNG.

    Args:
        app: Compiled LangGraph app
        registry: Dict mapping node names to functions (for file path extraction)
        path: Output file path
        theme: "dark" or "light"
        xray: Whether to show internal graph structure (including subgraph internals)
        show_file_paths: Whether to show file paths under node names
        debug: If True, prints the raw and final mermaid strings for inspection
    """
    try:
        # Get mermaid from graph
        mermaid = app.get_graph(xray=xray).draw_mermaid()

        if debug:
            print("=" * 60)
            print("RAW MERMAID (before transforms):")
            print("=" * 60)
            print(mermaid)
            print("=" * 60)

        # Inject file paths if requested
        if show_file_paths and registry:
            node_file_map = {}
            for name, fn in registry.items():
                module = getattr(fn, "__module__", None)
                if module:
                    parts = module.split(".")
                    node_file_map[name] = "/".join(parts[-2:]) + ".py"

            file_color = THEME_FILE_PATH if theme == "dark" else "#666666"
            mermaid = inject_file_labels(mermaid, node_file_map, file_color)

        # Save mermaid source before dark theme (GitHub handles theming natively)
        if save_mmd:
            mmd_path = path.replace(".png", ".mmd")
            with open(mmd_path, "w") as f:
                f.write(mermaid)
            print(f"Mermaid source saved to: {mmd_path}")

        # Apply dark theme modifications (only for dark mode)
        if theme == "dark":
            mermaid = apply_dark_theme(mermaid)

        if debug:
            print("FINAL MERMAID (after transforms):")
            print("=" * 60)
            print(mermaid)
            print("=" * 60)

        # Render and save
        png = asyncio.run(_render_mermaid_png(mermaid, theme=theme))
        with open(path, "wb") as f:
            f.write(png)
        print(f"Graph saved to: {path}")

    except Exception as e:
        print(f"Could not generate PNG: {e}")
