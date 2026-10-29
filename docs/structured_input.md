# Current-frame structured input

This optional DrivoR experiment concatenates structured scene tokens with the
existing image/LiDAR scene memory. It is disabled in the base `drivoR` config.
Select `agent=drivoR_structured` or set `agent.config.structured_input.enabled=true`
to enable it. No changes to the existing trajectory imitation, LK, or score losses
are required.

## Input provenance

**This is a privileged ground-truth-input ablation, not a sensor-only agent.**
Current actor annotations, the map API, and observed traffic-light flags come from
`Scene`. The agent advertises `requires_scene=true`, rejects sensor-only
`AgentInput` at inference, and is not eligible for the standard sensor-only
submission interface. No future annotations, future positions, PDM observation
cache, or metric-cache targets are used to construct these input features.

Only `frames[num_history_frames - 1]` is used for geometry and traffic lights.
Actor acceleration uses the same track in the preceding history frame, if
available. Previous velocities are rotated into the current ego axes before
finite differencing, using timestamps in microseconds. The actor states remain
current-time states; history is only used to estimate current acceleration.

## Fields

All positions use the current ego rear-axle frame, in metres. Rotations follow
the dataset's local actor headings. Velocity and acceleration are two-dimensional
components in current ego axes, not scalar magnitudes.

| Input | Values |
| --- | --- |
| Actor | `x, y, sin(yaw), cos(yaw), width, length, vx, vy, ax, ay, velocity_valid, acceleration_valid` |
| Actor type | vehicle, pedestrian, bicycle, static/other supported object |
| Map polyline | sampled `x,y` points plus semantic type |
| Traffic light | lane-connector entrance `x,y` plus `is_red` |

Source bounding boxes store `length,width`; the feature builder deliberately
outputs `width,length`. Invalid geometry is excluded. Missing/non-finite velocity
and unavailable acceleration are zero-filled with validity bits set to zero.
Acceleration is not inferred from future frames or from unrelated tracks.

The ten map types are lane centerline, connector centerline, lane left/right
boundary, connector left/right boundary, stop-line outline, crosswalk outline,
walkway outline, and intersection boundary. They are semantic geometry types,
not fabricated solid/dashed or paint-colour labels. Map geometry is transformed,
clipped to the ROI, split into bounded segments, and sampled using the existing
scene-geometry vectorization utilities.

The dataset supplies a binary red flag, so `is_red=false` means non-red as
represented by the dataset; it must not be interpreted as a four-state
red/yellow/green/unknown label. A light's location is the corresponding lane
connector's entrance, not a detected physical lamp position.

Default capacities are **128 actors, 256 map polylines with 20 ordered points each,
and 16 traffic lights**, in `[-32,-32,64,32]` metres. These are entity counts, not
semantic-type counts. Missing required map access raises an error; an agents-only
ablation does not require the map.

## Entity selection

All selection uses only current/history observations and map geometry. Tie breaks
use stable track/map identifiers. No generated trajectory, future GT, or PDM score
is used to choose inputs.

### Actors: 128 slots

`agent_selection=balanced` uses the following staged selection without repeating
tracks across stages:

| Stage | Budget | Ranking |
| --- | --- | --- |
| Near field | 64 | Distance between actor and ego bounding-box polygons |
| Approaching risk | Up to 32 | Smallest projected footprint gap, then time to closest approach |
| Spatial coverage | Remaining slots | Round-robin over eight angular sectors, nearest first in each sector |

The risk stage extrapolates constant relative velocity for at most five seconds
and considers actors with a projected gap of at most two metres. Closest-approach
time uses center-relative motion, and projected gap uses the oriented actor box
against the ego footprint. This is an approximate prioritization heuristic, not
the official responsibility-aware TTC metric. Missing actor or ego velocity
disables that actor's risk-stage priority, not its near-field/spatial eligibility.

When there are too few risk candidates, spatial selection fills the remainder.
When some sectors are empty, the other sectors continue filling. If fewer than
128 actors exist, unused positions remain masked padding. `agent_selection=nearest`
is available as a simpler footprint-distance ablation. Capacity overrides scale
the near/risk budgets to approximately one half and one quarter respectively.

### Map: 256 slots with type backfilling

The configured quota proportions are rescaled to the selected capacity. At the
default 256 map slots, the effective soft quotas in `MAP_TYPES` order are:

| Geometry type | Slots |
| --- | --- |
| Lane centerline | 48 |
| Connector centerline | 32 |
| Lane left boundary | 32 |
| Lane right boundary | 32 |
| Connector left boundary | 24 |
| Connector right boundary | 24 |
| Stop-line outline | 24 |
| Crosswalk outline | 16 |
| Walkway outline | 12 |
| Intersection boundary | 12 |

Choose within each type first, then fill unused quotas from all remaining
candidates. A missing type never reserves empty slots while another type has
available geometry. Changing `max_map_lines` proportionally rescales the quotas,
with deterministic largest-remainder allocation.

Within-type ranking favors geometry near ego and the nearest lane centerline,
with a penalty for segments wholly behind `x=-5 m`. Original map-object identities
are retained through clipping/splitting; exact sampled duplicates within a type
are removed. Each source's first two segments get an earlier priority round than
its extra segments. This is a soft diversity preference: extra segments can still
fill otherwise unused capacity. The reference lane is selected geometrically,
not from a future route or trajectory; this is not a full route-topology search.

The controls are `map_type_quotas`, `map_segments_per_source=2`, and
`map_selection=balanced`. `map_selection=nearest` provides a distance-only
ablation without quota allocation.

### Traffic lights: 16 slots

`light_selection=relevant` prioritizes connector entrances that are no more than
five metres behind ego, within eight metres of the reference centerline, and
whose start-to-end connector direction is not backward in ego x. If no reference
lane is available, lateral distance to the ego x axis is used. This is a local
relevance heuristic, not a guarantee that the light controls the planned route.

Relevant candidates come first, then entrance distance; a red flag breaks equal
distance ties, followed by connector ID. The remaining available candidates fill
unused slots. Duplicate statuses for the same connector produce one entity, with
red taking precedence on conflicting entries. `light_selection=nearest` disables
the relevance priority but still deduplicates connectors.

## Encoding and fusion

The actor MLP encodes its 12 numeric fields and adds an actor-type embedding.
Each polyline flattens its 20 ordered x/y samples to 40 values, passes through an
MLP, and adds a semantic-type embedding. This preserves sampled point order
instead of max-pooling points. Traffic lights have a separate three-input MLP.
Numeric inputs use fixed physical scales to keep magnitudes comparable.

There is **no pooling across actors, polylines, lights, or semantic types**. Each
entity is one token. With all modalities enabled, the encoder returns
`tokens: [B,400,tf_d_model]` and `valid: [B,400]`, ordered as 128 actor slots,
256 polyline slots, and 16 light slots. Empty positions have finite zero values
and are excluded using `key_padding_mask` in every decoder cross-attention
layer. Merely setting padding values to zero is not used as a masking substitute.

The fused attention memory is `[visual_scene_tokens, structured_tokens]`.
`structured_input.fusion=both` supplies it to generator and scorer. `gen` and
`scorer` restrict fusion to one branch. In scorer-only mode, structured inputs
cannot change the generated candidate set. Existing scene-geometry prediction,
when independently enabled, still reads the original visual memory.

Disabled mode adds no parameters or input requirements. Shared original model
weights and initialization RNG state are preserved even when the new encoder is
constructed, allowing seed-controlled comparisons. Existing strict checkpoint
loading is retained: baseline checkpoints cannot silently populate the new
encoder. Start a fresh run for the comparisons below; new structured checkpoints
must be reconstructed with the matching structured config. Earlier three-summary
token checkpoints are not architecture-compatible with this entity-token version
and must not be loaded as if they were trained entity encoders.

Changing only entity capacities does not change learned parameter shapes, so
earlier 64/64/16 entity-token checkpoints still load strictly. Increasing the
input budget nevertheless changes the experiment: record the evaluation capacity,
or override `max_agents=64` and `max_map_lines=64` to reproduce the earlier budget.

## Experiments

The new launcher reuses the existing B0 training settings: pretrained DINOv2,
fresh planning/scoring modules, seed 2, 10 epochs, four GPUs with batch size 16,
and the same LR/loss settings. It does not launch any training until invoked.

```bash
cd /mnt/workspace/roa7sgh/DrivoR
DRY_RUN=1 bash scripts/nav2_train_structured.sh

# Baseline: sensor-only DrivoR, with identical training settings
STRUCTURED_ENABLED=false EXPERIMENT_NAME=structured_b0 \
  bash scripts/nav2_train_structured.sh

# 128 actors, 256 polylines and 16 lights to generator and scorer
EXPERIMENT_NAME=structured_entity_gt_both bash scripts/nav2_train_structured.sh

# Branch ablations
STRUCTURED_FUSION=gen EXPERIMENT_NAME=structured_gt_gen \
  bash scripts/nav2_train_structured.sh
STRUCTURED_FUSION=scorer EXPERIMENT_NAME=structured_gt_scorer \
  bash scripts/nav2_train_structured.sh

# Actors only
EXPERIMENT_NAME=structured_gt_agents bash scripts/nav2_train_structured.sh \
  agent.config.structured_input.use_map=false \
  agent.config.structured_input.use_traffic_lights=false

# Map and traffic lights only
EXPERIMENT_NAME=structured_gt_map_lights bash scripts/nav2_train_structured.sh \
  agent.config.structured_input.use_agents=false

# Same entity architecture and capacities, simpler selection for comparison
EXPERIMENT_NAME=structured_entity_nearest bash scripts/nav2_train_structured.sh \
  agent.config.structured_input.agent_selection=nearest \
  agent.config.structured_input.map_selection=nearest \
  agent.config.structured_input.light_selection=nearest
```

`PY` or `PYTHON_BIN` selects the interpreter; the default is the verified system
Python at `/usr/local/bin/python3`. GPU, batch, worker, seed, loss, and background
settings follow `nav2_train_safety.sh`. The default submits a nohup job and prints
its log and PID. Use `BACKGROUND=0` for foreground execution. Existing scripts,
including user modifications to `nav2_train_scene_geometry.sh`, are not changed.

Use the same split, seeds, ranking weights, cache, training budget, and checkpoint
selection criterion for all runs. Compare `val/score`, `val/best_score`,
`val/lost_score`, NC/DAC/TTC/LK/progress/comfort, rather than training loss alone.
Repeat the baseline and best structured setting across multiple seeds before
attributing improvements to structured information.

These validation scores use DrivoR's training scorer; they are not full official
NAVSIM-v2 two-stage EPDMS. Only DrivoR was modified for this experiment. The sibling
`navsim-v2` copied evaluation agents are not adapted here to load or evaluate
structured checkpoints. Current-GT results are an oracle-input upper-bound study,
not a deployable sensor-only benchmark claim.

## Cache and checks

Existing training metric caches and scene-geometry target caches are unchanged
and need no rebuild. The launcher defaults to online features (`cache_path=null`).
If using feature caches, compute the additional `structured_current_v2_<hash>`
feature file with the new builders. Earlier image-only or v1 structured feature
caches do not encode the current selection rules. Cache identity includes
capacities, ROI, sampling, modality choices, risk settings, map quotas, and
selection modes, but not generator/scorer fusion mode. Online, offline caching,
and cache-only loading use the same fields.

```bash
python -m pytest -q tests/agents/test_structured_input.py tests/agents/test_training_launcher.py
bash -n scripts/nav2_train_structured.sh
```

Tests cover per-entity token counts, permutation equivariance, ordered polylines,
attention padding exclusion, no future access, causal acceleration, risk/spatial
selection, map quotas and backfill, source diversity, light relevance and
deduplication, cache round trips, both decoder memories, gradients, unchanged
disabled behavior, and launcher configuration.
Three real current/history-only scenes and a full DINOv2 mixed-precision GPU
forward/backward were checked. No long training run or numerical performance
improvement is claimed by these smoke tests.