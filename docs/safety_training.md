# NAV2 safety training

The objective keeps the original trajectory imitation and score losses and adds
`dac_weight * dac_geometry_loss + collision_weight * collision_geometry_loss`.
Both new terms supervise every candidate in the final refinement layer, before
the discrete scorer selection. Default agent configuration disables both terms;
the launcher also defaults to B0 with both weights zero. Enable them explicitly
for B2; the example below uses weights 1.0 and 1.0. Tune these weights against real
DAC/NC, progress, comfort and candidate diversity, not loss alone.

Each safety term averages the per-trajectory penalties over all candidates
(64 by default) and the batch. Safe candidates contribute zero but remain in the
candidate denominator. One violating candidate among 64 contributes its penalty
divided by 64; its direct safety gradient is also divided by 64 before weighting.
Candidates satisfying the safety margins have zero direct safety gradient. An
all-safe sample contributes a differentiable zero. Positive penalties include
insufficient safety margin or leaving the DAC raster, not just a failure of the
official DAC/NC metric. Within-trajectory time, footprint and obstacle reductions
are unchanged.

## Geometry and targets

Safety targets come from the existing compressed training metric cache. The DAC
field uses the same drivable polygon layers as the original training scorer.
Its positive values indicate drivable space; negative values indicate outside.
The field is rasterized at 0.5 m resolution and expressed in the initial rear-axle
frame. A bounded per-worker cache avoids retaining the entire dataset in memory.

The vehicle length, width and rear-axle-to-center offset come from cached vehicle
parameters. Fifteen rectangular footprint samples, including the corners and
interior, query the field by differentiable bilinear interpolation. Squared hinge
penalties enforce the configured margin; mean and maximum penalties are combined.
Leaving the raster adds a restoring position penalty. Missing map coverage is
not inferred as safe. Only geometry present in the metric cache is available;
changing raster bounds does not expand the underlying map coverage.

Collision supervision uses all future obstacle footprints in cached observations,
not just objects that collided with an existing proposal. Red-light pseudo-objects
and tracks already marked collided in the cache are excluded, matching the scorer
exclusions. Obstacles are represented by enclosing oriented rectangles and masked
when absent. A separating-axis gap gives a differentiable, conservative collision
surrogate. This is not the exact responsibility-aware NC metric. Actor projections
are processed in chunks to bound peak memory. Mean and worst penalties are combined.

Predicted positions and unwrapped headings are interpolated from the initial pose
to observation timestamps (normally 0.1 s). No gradient is detached on this loss
path. Targets remain constant. Finite spatial/temporal sampling and soft penalties
do not provide a hard safety guarantee. Map/actor ground truth is training-only;
inference inputs and architecture are unchanged.

## Launch

The launcher follows `scorer-grad`'s `scripts/nav1_train_frozen_backbones.sh` path
and GPU conventions. Its default is experiment B0: pretrained DINOv2 plus original
trajectory imitation and score losses, without the additional geometry losses.
It does not load a DrivoR
checkpoint (`agent.checkpoint_path=''`) or resume an old training run.
Original DINOv2 weights are loaded independently from
`weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors` and stay frozen.
LoRA, neck, scene embeddings, ego encoder, proposal embeddings, trajectory decoder,
trajectory heads, position embedding, scorer attention and score heads start from
new initialization. All trainable parameters use the same default learning rate
(`PERCEPTION_LR_SCALE=1.0`). The base learning rate is 0.0002 (scaled by global batch
size), and training runs for 10 epochs. No auxiliary semantic map head is added.
The default is four GPUs with batch size 16 per GPU, seed 2, prefetch factor 1 and
16 DataLoader workers per rank. Gradient clipping is disabled (`gradient_clip_val=0`)
and persistent workers use the DataLoader default (`false`), matching the README's
NAV2 recipe. This changes launcher parameters only, not installed library versions.

```bash
DRY_RUN=1 bash scripts/nav2_train_safety.sh
bash scripts/nav2_train_safety.sh
LOG_FILE=exp/logs/xx.log bash scripts/nav2_train_safety.sh
tail -f exp/logs/xx.log
BACKGROUND=0 GPU_IDS=0 BATCH_SIZE=2 DATALOADER_WORKERS=2 USE_RAY=false \
  bash scripts/nav2_train_safety.sh trainer.params.fast_dev_run=true

# B2: same initialization and training settings, with additional safety losses
DAC_WEIGHT=1 COLLISION_WEIGHT=1 EXPERIMENT_NAME=nav2_safety_b2 \
  bash scripts/nav2_train_safety.sh
```

Training is submitted with `nohup` by default. The launcher prints the submitted
PID and log path, creates the log directory, and redirects stdout and stderr to
`exp/logs/<experiment>_<timestamp>.log`. Standard input is disconnected and Python
output is unbuffered, so training can continue after closing the terminal.
Set `LOG_FILE` for a specific filename; reusing that filename overwrites the old
console log. These are console logs, not TensorBoard event files. `BACKGROUND=0`
runs in the foreground. Hydra configuration/help queries stay in the foreground
automatically, and `DRY_RUN=1` only prints the command without creating log files.

Defaults match the NAV1 branch: data under
`/mnt/workspace/hru4sgh/NAVSIM/dataset`, metric cache under
`DrivoR/dataset/train_metric_cache_navtrain`. A DrivoR checkpoint under `weights/`
is not required. Override `DATA_ROOT`,
`TRAIN_METRIC_CACHE_PATH`, `GPU_IDS`, `BASE_LR`, `PERCEPTION_LR_SCALE`, `DAC_WEIGHT`,
`COLLISION_WEIGHT`, `DAC_MARGIN`, `COLLISION_MARGIN` or `SDF_RESOLUTION` as needed.
Additional arguments are forwarded to Hydra. The script sets `DRIVOR_NO_RESUME=1`
and `train_ckpt_path=null` to prevent accidental restoration of an old experiment.
New Lightning checkpoints are saved under `exp/ke/nav2_safety_b0/...` by default,
not `weights/`. `EXPERIMENT_NAME` selects a separate output directory for B2.
Explicit Hydra overrides can load or resume a checkpoint, but that is no longer
the fresh-initialization experiment. Compare B0 and B2 using the same seeds,
data and evaluation protocol; the published DrivoR score is an additional reference.

This is the NAV2 model training recipe in DrivoR's NAV1-based runtime, as in its
README. It preserves the other training metric definitions and selection weights;
it does not turn the scorer into the full NAVSIM-v2 EPDMS evaluation protocol. The current
agent additionally supervises a lane-keeping head and includes LK with weight 2.0
in the training aggregate and default candidate ranking; see
[Lane keeping](lane_keeping.md) for cache and legacy-checkpoint compatibility.
Run official NAV2 evaluation in the separate `navsim-v2` checkout.

```bash
python -m unittest discover -s tests/agents -p 'test_*.py' -v
```