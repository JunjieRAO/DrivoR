# Lane keeping

DrivoR computes the NAVSIM-v2 lane-keeping (LK) label online for every simulated
candidate and predicts it with a seventh score head. The head uses the existing
scorer features and does not require additional inference inputs.

## Metric and cache

LK uses the vehicle bounding-box center's distance to the cached route centerline.
The default deviation limit is 0.5 m and the horizon window is 2.0 s. The score is
zero when the number of consecutive offending samples reaches
`ceil(2.0 / interval_length)`; otherwise it is one. A distance exactly equal to
0.5 m passes. Intersection samples are skipped without incrementing or resetting
the counter, matching the official NAVSIM-v2 implementation.

The existing compressed training metric cache already contains the centerline and
map layers needed for LK. It does not need to be rebuilt. Candidate-dependent LK
labels are computed after proposal simulation, not stored as fixed cache labels.
Both raw Shapely centerlines in legacy training caches and `PDMPath` centerlines
are accepted.
Saved inference candidates or scores must be regenerated after changing the model
or ranking settings, according to the evaluation workflow's cache policy.

The score-label columns are now
`[NC, DAC, EP, TTC, comfort, DDC, LK, total]`. Existing submetric positions remain
unchanged, and the aggregate score remains the last column.

## Training and ranking

The new `scorer.pred_score.lane_keeping` MLP matches the other heads:
`tf_d_model -> tf_d_ffn -> 1`, with a ReLU hidden activation. Its BCE loss has
weight 1.0 in `final_score_loss`; training logs include `train/lk_loss` and
validation logs include `val/lk`. The optional secondary scorer also emits seven
values when `double_score=true`.

LK has weight 2.0 in the training scorer's weighted aggregate. Predicted LK is
also added to the TTC/EP/comfort weighted sum used for candidate selection, with
`agent.config.lk=2.0` by default. It is not a multiplicative compliance gate.
Setting `agent.config.lk=0` disables only its predicted ranking contribution;
LK labels and the BCE loss remain enabled.

This adds LK to the existing training metrics. It does not replace the remaining
NAV1-based training scorer with the complete NAVSIM-v2 EPDMS protocol. Official
NAVSIM-v2 evaluation still runs in the separate `navsim-v2` checkout. Both its
base DrivoR and scene-geometry evaluation agents support the added head.

## Checkpoints

New checkpoints include all seven heads and load strictly. A complete legacy
six-head checkpoint can be loaded through `agent.checkpoint_path` only when
`agent.config.lk=0`. The missing LK head starts from fresh initialization and a
warning is emitted. After fine-tuning it, enable LK ranking for evaluation with
`agent.config.lk=2.0`. Missing weights outside this new head, partially missing LK
weights, unexpected keys, and shape mismatches remain errors.

For example, warm-start a new run from an old checkpoint without resuming its
optimizer state:

```bash
bash scripts/nav2_train_safety.sh \
  experiment_name=nav2_lk_finetune \
  agent.checkpoint_path=/path/to/old_checkpoint.ckpt \
  agent.config.lk=0
```

Fresh training requires no special override. Exact Lightning resume of a legacy
six-head checkpoint is not supported; start a new fine-tuning run instead.
The legacy compatibility path assumes the default `double_score=false`.
Old `double_score=true` checkpoints also change output shape and are not migrated.

## Evaluation launchers

The local NAVSIM-v2 base and GEN-SCORER nohup launchers default `LK_WEIGHT=0`
because their default checkpoints predate the new head. For a newly trained
seven-head GEN-SCORER checkpoint, run from the separate `navsim-v2` checkout:

```bash
CHECKPOINT_PATH=/path/to/new_gen_scorer.ckpt LK_WEIGHT=2 \
  bash scripts/evaluation/run_drivor_gen_scorer_nav2_nohup.sh
```

Direct Python evaluation uses `agent.config.lk` instead of the launcher environment
variable. Candidate diagnostic files include predicted LK when the head is present
and record `predicted_metric_names` alongside the probability columns. Legacy
six-head outputs remain identifiable as six columns without `lane_keeping`.

## Checks

```bash
python -m pytest -q tests/agents/test_lane_keeping.py \
  tests/agents/test_checkpoint_initialization.py tests/agents/test_safety_loss.py
```