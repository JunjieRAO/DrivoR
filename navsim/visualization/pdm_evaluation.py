import logging
import re
from pathlib import Path
from typing import Any, Dict, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from navsim.visualization.bev import add_trajectory_to_bev_ax
from navsim.visualization.plots import plot_bev_frame

logger = logging.getLogger(__name__)


BASELINE_TRAJECTORY_CONFIG: Dict[str, Any] = {
    "line_color": "#d62728",
    "line_color_alpha": 1.0,
    "line_width": 3.0,
    "line_style": "-",
    "marker": "o",
    "marker_size": 5,
    "marker_edge_color": "black",
    "zorder": 5,
}

ORACLE_TRAJECTORY_CONFIG: Dict[str, Any] = {
    "line_color": "#1f77b4",
    "line_color_alpha": 1.0,
    "line_width": 3.0,
    "line_style": "--",
    "marker": "s",
    "marker_size": 5,
    "marker_edge_color": "black",
    "zorder": 4,
}

VISUALIZATION_INDEX_COLUMNS = (
    "token",
    "failure_reason",
    "selected_proposal_idx",
    "best_proposal_idx",
    "no_at_fault_collisions",
    "driving_direction_compliance",
    "score",
    "best_no_at_fault_collisions",
    "best_driving_direction_compliance",
    "best_score",
    "image",
)


def select_failure_visualizations(results: pd.DataFrame, num_scenarios: int) -> pd.DataFrame:
    """Select baseline NC/DDC failures by ascending ground-truth score."""
    if num_scenarios < 0:
        raise ValueError(f"Expected a non-negative visualization count, got {num_scenarios}")

    required_columns = {
        "token",
        "valid",
        "no_at_fault_collisions",
        "driving_direction_compliance",
        "score",
    }
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(f"Missing visualization result columns: {sorted(missing_columns)}")

    eligible = results[
        results["valid"].astype(bool)
        & (
            results["no_at_fault_collisions"].eq(0)
            | results["driving_direction_compliance"].eq(0)
        )
    ].copy()
    eligible["failure_reason"] = eligible.apply(_failure_reason, axis=1)
    return eligible.sort_values(["score", "token"], kind="stable").head(num_scenarios)


def render_failure_visualizations(
    results: pd.DataFrame,
    scene_loader: Any,
    predictions: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    """Render selected baseline failures against their oracle trajectories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for sequence, (_, row) in enumerate(results.iterrows(), start=1):
        token = str(row["token"])
        figure = None
        try:
            prediction = predictions[token]
            selected_idx = int(row["selected_proposal_idx"])
            best_idx = int(row["best_proposal_idx"])
            scene = scene_loader.get_scene_from_token(token)
            frame_idx = scene.scene_metadata.num_history_frames - 1
            figure, axis = plot_bev_frame(scene, frame_idx)

            add_trajectory_to_bev_ax(
                axis,
                prediction["proposals"][selected_idx],
                BASELINE_TRAJECTORY_CONFIG,
            )
            axis.lines[-1].set_label("Baseline selected")
            add_trajectory_to_bev_ax(
                axis,
                prediction["proposals"][best_idx],
                ORACLE_TRAJECTORY_CONFIG,
            )
            axis.lines[-1].set_label("Oracle")
            axis.legend(loc="upper right")
            axis.set_title(
                f"{token} | {row['failure_reason']}\n"
                f"baseline [{selected_idx}] {row['score']:.3f} | "
                f"oracle [{best_idx}] {row['best_score']:.3f}"
            )

            safe_token = re.sub(r"[^A-Za-z0-9_.-]", "_", token)
            image_name = f"{sequence:03d}_{safe_token}_{row['failure_reason']}.png"
            figure.savefig(output_dir / image_name, bbox_inches="tight", dpi=150)
            index_rows.append(
                {
                    "token": token,
                    "failure_reason": row["failure_reason"],
                    "selected_proposal_idx": selected_idx,
                    "best_proposal_idx": best_idx,
                    "no_at_fault_collisions": row["no_at_fault_collisions"],
                    "driving_direction_compliance": row["driving_direction_compliance"],
                    "score": row["score"],
                    "best_no_at_fault_collisions": row["best_no_at_fault_collisions"],
                    "best_driving_direction_compliance": row["best_driving_direction_compliance"],
                    "best_score": row["best_score"],
                    "image": image_name,
                }
            )
        except Exception:
            logger.exception("Failed to visualize evaluation token %s", token)
        finally:
            if figure is not None:
                plt.close(figure)

    index = pd.DataFrame(index_rows, columns=VISUALIZATION_INDEX_COLUMNS)
    index.to_csv(output_dir / "index.csv", index=False)
    return index


def _failure_reason(row: pd.Series) -> str:
    nc_failed = row["no_at_fault_collisions"] == 0
    ddc_failed = row["driving_direction_compliance"] == 0
    if nc_failed and ddc_failed:
        return "nc0_ddc0"
    return "nc0" if nc_failed else "ddc0"