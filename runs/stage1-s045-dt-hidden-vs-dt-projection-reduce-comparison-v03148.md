# Stage 1 Phase Timing Comparison

- Baseline: `runs/stage1-s045-dt-hidden-reduce-lora-merged-mamba130m-v03146.json`
- Candidate: `runs/stage1-s045-dt-projection-reduce-lora-merged-mamba130m-v03148.json`
- Eval speedup: `1.228`

| phase | baseline s | candidate s | delta s | speedup |
|---|---:|---:|---:|---:|
| `layer_0.dt_rank_projection` | 107.505 | 7.320 | 100.185 | 14.687 |
| `layer_0.gate_projection` | 159.283 | 150.216 | 9.067 | 1.060 |
| `layer_0.conv_projection` | 160.641 | 159.319 | 1.322 | 1.008 |
| `layer_0.output_projection` | 156.480 | 155.640 | 0.840 | 1.005 |
| `layer_0.dt_hidden_projection` | 3.767 | 3.213 | 0.554 | 1.173 |
| `layer_0.rank_silu_polynomial` | 5.463 | 5.165 | 0.298 | 1.058 |
| `layer_0.gate_silu_polynomial` | 3.465 | 3.366 | 0.099 | 1.029 |
| `layer_0.decay_state_major_polynomial` | 3.616 | 3.527 | 0.089 | 1.025 |
