from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from navsim.common.dataclasses import Scene, Trajectory
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.plots import configure_ax


SAFE_CONFIG = {
    "line_color": "#2e9d57",
    "line_color_alpha": 0.48,
    "line_width": 0.55,
    "line_style": "-",
    "marker": "o",
    "marker_size": 1.6,
    "marker_edge_color": "none",
    "zorder": 7,
}
UNSAFE_CONFIG = {**SAFE_CONFIG, "line_color": "#d84a4a"}
SELECTED_CONFIG = {
    **SAFE_CONFIG,
    "line_color": "#1769c2",
    "line_color_alpha": 1.0,
    "line_width": 1.25,
    "line_style": "--",
    "marker_size": 1.9,
    "zorder": 11,
}
ORACLE_CONFIG = {
    **SAFE_CONFIG,
    "line_color": "#8a46a3",
    "line_color_alpha": 1.0,
    "line_width": 1.35,
    "marker_size": 1.9,
    "zorder": 10,
}
GT_CONFIG = {
    **SAFE_CONFIG,
    "line_color": "#111111",
    "line_color_alpha": 1.0,
    "line_width": 1.25,
    "marker_size": 1.8,
    "zorder": 11,
}


def proposal_is_safe(result: Dict[str, Any]) -> bool:
    """Return whether real PDM safety metrics NC, DAC, and TTC all pass."""
    return bool(
        np.isclose(result["no_at_fault_collisions"], 1.0)
        and np.isclose(result["drivable_area_compliance"], 1.0)
        and np.isclose(result["time_to_collision_within_bound"], 1.0)
    )


def selected_is_unsafe(detail: Dict[str, Any]) -> bool:
    selected_idx = int(detail["selected_proposal_idx"])
    result = detail["results"][selected_idx]
    return bool(
        np.isclose(result["no_at_fault_collisions"], 0.0)
        or np.isclose(result["drivable_area_compliance"], 0.0)
        or np.isclose(result["time_to_collision_within_bound"], 0.0)
    )


def render_proposal_safety_scene(
    scene: Scene,
    token: str,
    proposals: Sequence[Trajectory],
    proposal_results: Sequence[Dict[str, Any]],
    selected_idx: int,
    oracle_idx: int,
    output_path: Path,
) -> None:
    """Render a real NAVSIM scene with all proposals and safety overlays."""
    if len(proposals) != len(proposal_results):
        raise ValueError(
            f"Proposal/result count mismatch for {token}: {len(proposals)} != {len(proposal_results)}"
        )
    if not 0 <= selected_idx < len(proposals) or not 0 <= oracle_idx < len(proposals):
        raise IndexError(f"Selected/oracle proposal index is invalid for token {token}")

    frame_idx = scene.scene_metadata.num_history_frames - 1
    figure, axis = plt.subplots(figsize=(8.2, 9.0), dpi=180)
    add_configured_bev_on_ax(axis, scene.map_api, scene.frames[frame_idx])

    for proposal, result in zip(proposals, proposal_results):
        add_trajectory_to_bev_ax(
            axis,
            proposal,
            SAFE_CONFIG if proposal_is_safe(result) else UNSAFE_CONFIG,
        )
    add_trajectory_to_bev_ax(axis, proposals[oracle_idx], ORACLE_CONFIG)
    add_trajectory_to_bev_ax(axis, scene.get_future_trajectory(), GT_CONFIG)
    add_trajectory_to_bev_ax(axis, proposals[selected_idx], SELECTED_CONFIG)

    selected_result = proposal_results[selected_idx]
    axis.set_xlim(32.0, -32.0)
    axis.set_ylim(-20.0, 50.0)
    axis.set_aspect("equal")
    configure_ax(axis)
    axis.set_title(
        f"{token}\n"
        f"selected={selected_idx}  oracle={oracle_idx}  "
        f"NC={selected_result['no_at_fault_collisions']:.2f}  "
        f"DAC={selected_result['drivable_area_compliance']:.2f}  "
        f"TTC={selected_result['time_to_collision_within_bound']:.2f}",
        fontsize=9,
    )
    axis.legend(
        handles=[
            Line2D([0], [0], color=SAFE_CONFIG["line_color"], lw=1.2, label="Safe proposal"),
            Line2D([0], [0], color=UNSAFE_CONFIG["line_color"], lw=1.2, label="Unsafe proposal"),
            Line2D([0], [0], color=SELECTED_CONFIG["line_color"], lw=1.5, ls="--", label="Selected"),
            Line2D([0], [0], color=ORACLE_CONFIG["line_color"], lw=1.5, label="Oracle"),
            Line2D([0], [0], color=GT_CONFIG["line_color"], lw=1.5, label="GT"),
        ],
        loc="lower right",
        fontsize=7,
        framealpha=0.92,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
