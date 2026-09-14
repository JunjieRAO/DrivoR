"""Deterministic control-space search and selection, independent of NAVSIM imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Config:
    seed: int = 20260914
    population: int = 32
    generations: int = 5
    archive_size: int = 32
    min_gain: float = 0.005
    collision_margin: float = 0.25
    road_margin: float = 0.10
    gt_path_margin: float = 0.75
    min_interval: float = 0.0125
    dt: float = 0.1
    horizon: float = 4.0

    def __post_init__(self):
        if not (4 <= self.population <= 128 and 1 <= self.generations <= 20):
            raise ValueError("population must be 4..128; generations must be 1..20")
        if self.dt != 0.1 or self.horizon != 4.0:
            raise ValueError("This engineering runner only supports V1: 40 x 0.1 s")
        if not (1 <= self.archive_size <= 32 and 0 < self.min_gain <= 1):
            raise ValueError("invalid archive_size/min_gain")
        if min(self.collision_margin, self.road_margin, self.gt_path_margin) <= 0:
            raise ValueError("safety margins must be positive")
        if not (0 < self.min_interval <= self.dt):
            raise ValueError("invalid minimum interval")


KNOTS = np.array([0., .5, 1., 2., 3., 4.])
BOUNDS = np.array([1.] * 5 + [.03] * 5)
STD = np.array([.25] * 5 + [.008] * 5)
FLOOR = np.array([.03] * 5 + [.001] * 5)


def basis() -> np.ndarray:
    return np.stack([np.interp(np.arange(40) * .1, KNOTS, np.eye(6)[i])
                     for i in range(1, 6)], axis=1)


def residual(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (10,) or not np.isfinite(theta).all() or np.any(abs(theta) > BOUNDS + 1e-12):
        raise ValueError("control residual is outside the fixed 10-dimensional bounds")
    return np.stack([basis() @ theta[:5], basis() @ theta[5:]], axis=1)


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], "little")


def candidate_id(poses: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(poses, dtype="<f8").tobytes()).hexdigest()[:20]


@dataclass
class Candidate:
    id: str
    poses: np.ndarray
    reference: np.ndarray
    executed: np.ndarray
    metrics: dict
    gate: dict
    generated: np.ndarray | None = None
    controls: np.ndarray | None = None
    theta: np.ndarray | None = None
    island: str = "GT"
    generation: int = -1
    # Partial engineering checks are intentionally not a training certificate.
    export_certified: bool = False

    @property
    def passed(self) -> bool:
        return bool(self.gate["checked_nominal_pass"])

    @property
    def score(self) -> float:
        return float(self.metrics["score"])

    def record(self) -> dict:
        return {"id": self.id, "island": self.island, "generation": self.generation,
                "metrics": self.metrics, "gate": self.gate,
                "export_certified": False, "theta": None if self.theta is None else self.theta.tolist()}


def quality(c: Candidate):
    return (-c.score, -c.gate.get("clearance_lower_bound", 0.),
            c.gate.get("max_jerk", float("inf")),
            float(np.linalg.norm(c.theta / BOUNDS)) if c.theta is not None else 0., c.id)


def truncated_sample(rng, mean, std, count):
    out = rng.normal(mean, std, (count, 10))
    fallback_count = 0
    for _ in range(100):
        mask = abs(out) > BOUNDS
        if not mask.any():
            return out, fallback_count
        new = rng.normal(mean, std, out.shape)
        out[mask] = new[mask]
    mask = abs(out) > BOUNDS
    fallback_count = int(mask.sum())
    out[mask] = np.broadcast_to(mean, out.shape)[mask]
    return out, fallback_count


def search(config: Config, token: str, gt: Candidate, seeds: dict[str, np.ndarray],
           evaluate: Callable[[np.ndarray], Candidate], on_candidate=None):
    """Search only against implemented nominal checks; final certification remains pending."""
    all_candidates: dict[str, Candidate] = {}
    theta_cache: dict[bytes, Candidate] = {}
    archive: dict[str, Candidate] = {}
    trace = []
    if gt.score + config.min_gain > 1. + 1e-12:
        return [], [], [{"status": "gt_upper_bound", "score": gt.score}]
    for island, initial in seeds.items():
        mean, std = initial.copy(), STD.copy()
        best = None
        for generation in range(config.generations):
            rng = np.random.default_rng(stable_seed(config.seed, token, island, generation))
            points, fallback = truncated_sample(rng, mean, std, config.population)
            points[0] = mean
            points[1] = best.theta if best is not None else initial
            batch = []
            for theta in points:
                key = np.ascontiguousarray(theta, dtype="<f8").tobytes()
                if key in theta_cache:
                    c = theta_cache[key]
                else:
                    c = evaluate(theta)
                    c.theta, c.island, c.generation = theta.copy(), island, generation
                    theta_cache[key] = c
                    all_candidates.setdefault(c.id, c)
                    if on_candidate:
                        on_candidate(c)
                batch.append((theta, c))
                if c.passed and c.score >= gt.score + config.min_gain:
                    archive.setdefault(c.id, c)
            feasible = [(t, c) for t, c in batch if c.passed]
            if feasible:
                elites = sorted(feasible, key=lambda p: quality(p[1]))[:4]
                values = np.stack([p[0] for p in elites])
                mean = .5 * mean + .5 * values.mean(axis=0)
                if len(elites) >= 4:
                    std = np.maximum(FLOOR, .5 * std + .5 * values.std(axis=0))
                else:
                    std = np.maximum(FLOOR, .8 * std)
                current = elites[0][1]
                if best is None or quality(current) < quality(best):
                    best = current
            else:
                # Unknown data must never masquerade as low violation.
                known = [(t, c) for t, c in batch if c.gate.get("data_verifiable", False)]
                if known:
                    elites = sorted(known, key=lambda p: tuple(p[1].gate["violation"]))[:4]
                    mean = .5 * initial + .5 * np.mean([p[0] for p in elites], axis=0)
                else:
                    mean = initial.copy()
                std = np.maximum(FLOOR, .5 * std)
            trace.append({"island": island, "generation": generation,
                          "feasible": len(feasible), "population": len(batch),
                          "best_score": None if best is None else best.score,
                          "sampler_fallback": fallback, "evaluations": len(theta_cache)})
    # Island quotas protect small modes; archive contains all generations.
    keep = {}
    ordered = sorted(archive.values(), key=quality)
    for island in seeds:
        for c in [c for c in ordered if c.island == island][:4]:
            if len(keep) < config.archive_size:
                keep[c.id] = c
    for c in ordered:
        if len(keep) == config.archive_size:
            break
        keep[c.id] = c
    trace.append({"improving_before_cap": len(ordered), "after_cap": len(keep),
                  "budget_truncated": len(ordered) - len(keep)})
    return list(all_candidates.values()), sorted(keep.values(), key=quality), trace


def trajectory_distance(a: Candidate, b: Candidate) -> float:
    """Engineering distance; anchor-time/stop categories remain a later audit gate."""
    x, y = a.executed, b.executed
    xy = np.sqrt(np.mean(np.sum((x[:, :2] - y[:, :2]) ** 2, axis=1)))
    speed = np.sqrt(np.mean((x[:, 3] - y[:, 3]) ** 2))
    mask = (abs(x[:, 3]) >= .5) & (abs(y[:, 3]) >= .5)
    heading = np.arctan2(np.sin(x[:, 2] - y[:, 2]), np.cos(x[:, 2] - y[:, 2]))
    angle = np.sqrt(np.mean(heading[mask] ** 2)) if mask.any() else 0.
    return float(max(xy / .75, speed / .75, angle / np.deg2rad(5)))


def representatives(candidates, gt, limit=4):
    """Complete-link groups; select actual members, then enforce pairwise separation."""
    clusters = []
    for c in sorted(candidates, key=quality):
        group = next((g for g in clusters if all(trajectory_distance(c, x) < 1 for x in g)), None)
        if group is None:
            clusters.append([c])
        else:
            group.append(c)
    chosen = []
    for c in sorted([min(g, key=quality) for g in clusters], key=quality):
        if all(trajectory_distance(c, x) >= 1 for x in [gt] + chosen):
            chosen.append(c)
            if len(chosen) == limit:
                break
    return chosen
