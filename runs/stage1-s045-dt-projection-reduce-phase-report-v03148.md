# Stage 1 Phase Timing Report

- Source: `runs/stage1-s045-dt-projection-reduce-lora-merged-mamba130m-v03148.json`
- Next bottleneck: `layer_0.conv_projection`

| phase | seconds | eval fraction | rotations | ct-pt | ct-ct |
|---|---:|---:|---:|---:|---:|
| `layer_0.conv_projection` | 159.319 | 0.323 | 71 | 2303 | 0 |
| `layer_0.output_projection` | 155.640 | 0.315 | 98 | 2303 | 0 |
| `layer_0.gate_projection` | 150.216 | 0.304 | 71 | 2303 | 0 |
| `layer_0.dt_rank_projection` | 7.320 | 0.015 | 575 | 96 | 0 |
| `layer_0.rank_silu_polynomial` | 5.165 | 0.010 | 0 | 0 | 14 |
| `layer_0.decay_state_major_polynomial` | 3.527 | 0.007 | 0 | 0 | 5 |
| `layer_0.gate_silu_polynomial` | 3.366 | 0.007 | 0 | 0 | 8 |
| `layer_0.dt_hidden_projection` | 3.213 | 0.007 | 575 | 96 | 0 |
| `layer_0.dynamic_b_projection` | 1.151 | 0.002 | 191 | 32 | 0 |
| `layer_0.dynamic_c_projection` | 1.135 | 0.002 | 191 | 32 | 0 |
| `layer_0.dynamic_b_broadcast` | 1.049 | 0.002 | 191 | 16 | 0 |
| `layer_0.dynamic_c_broadcast` | 1.037 | 0.002 | 191 | 16 | 0 |
