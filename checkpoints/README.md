# GR3 Mini Selected Checkpoints

This directory stores selected GR3 Mini trace-tracking checkpoints outside the ignored `logs/` tree.

`play.py` loads checkpoints from `logs/gr3mini_trace/<load_run>/model_<checkpoint>.pt`, so restore a selected checkpoint into a temporary log run before playing it.

## Files

| Branch | Config | Checkpoint |
| --- | --- | --- |
| `master` | future 5 frames, baseline, no randomization | `gr3mini_trace__master__future5__baseline__Jun23_11-43-52__model_7000.pt` |
| `randomlize` | future 5 frames, light randomization | `gr3mini_trace__randomlize__future5__light_rand__Jun23_13-48-13__model_10000.pt` |
| `ablation` | future 5 frames, `lr=1e-4`, `entropy_coef=0.02` | `gr3mini_trace__ablation__future5__lr1e-4_entropy0p02__Jun23_16-26-42__model_6800.pt` |
| `future-steps-1` | future 1 frame, baseline | `gr3mini_trace__future-steps-1__future1__baseline__Jun23_18-47-15__model_10000.pt` |
| `future-steps-0` | no future frames, baseline, partial run | `gr3mini_trace__future-steps-0__future0__baseline__Jun24_09-44-11__model_600.pt` |
| `gr3-turning-trace-split` | GR3 turning JSON train split, baseline before policy collapse | `gr3mini_trace__gr3-turning-trace-split__turning_json__baseline__Jun26_15-14-34__model_4800.pt` |
| `gr3-turning-trace-split` | GR3 turning JSON train split, `clip_actions=2.0`, light randomization, resumed from 4600 | `gr3mini_trace__gr3-turning-trace-split__turning_json__clip2_light_rand__Jun26_17-42-35__model_6400.pt` |

## Restore Example

Example for the `master` checkpoint:

```bash
cd /home/fftai/mjq/legged_gym/legged_gym

mkdir -p logs/gr3mini_trace/selected_master_future5
cp checkpoints/gr3mini_trace__master__future5__baseline__Jun23_11-43-52__model_7000.pt \
  logs/gr3mini_trace/selected_master_future5/model_7000.pt

conda run --no-capture-output -n legged_gym_py38 python -u legged_gym/scripts/play.py \
  --task=gr3mini_trace \
  --load_run=selected_master_future5 \
  --checkpoint=7000 \
  --num_envs=1
```

For `future-steps-1` and `future-steps-0`, switch to the matching branch before playing because the observation dimension changes.

For the turning checkpoints, switch to `gr3-turning-trace-split`. The `model_4800` checkpoint is from the best stable region before the original run collapsed. The `model_6400` checkpoint is from the resume run after enabling action clipping and light randomization; it was saved from local resume step 1800 and its checkpoint metadata has been corrected to `iter=6400`.

Note: some intermediate periodic checkpoints keep `iter=0` inside the checkpoint metadata even though the filename is `model_<iteration>.pt`. They are fine for `play.py`, but prefer final-run checkpoints for resume training.
