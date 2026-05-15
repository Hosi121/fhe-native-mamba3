# Stage 1 Phase Timing Report

- Source: `runs/stage1-s045-dt-hidden-reduce-lora-merged-mamba130m-v03146.json`
- Next bottleneck: `layer_0.conv_projection`

| phase | seconds | eval fraction | rotations | ct-pt | ct-ct |
|---|---:|---:|---:|---:|---:|
| `layer_0.conv_projection` | 160.641 | 0.265 | 71 | 2303 | 0 |
| `layer_0.gate_projection` | 159.283 | 0.263 | 71 | 2303 | 0 |
| `layer_0.output_projection` | 156.480 | 0.258 | 98 | 2303 | 0 |
| `layer_0.dt_rank_projection` | 107.505 | 0.177 | 79 | 1583 | 0 |
| `layer_0.rank_silu_polynomial` | 5.463 | 0.009 | 0 | 0 | 14 |
| `layer_0.dt_hidden_projection` | 3.767 | 0.006 | 575 | 96 | 0 |
| `layer_0.decay_state_major_polynomial` | 3.616 | 0.006 | 0 | 0 | 5 |
| `layer_0.gate_silu_polynomial` | 3.465 | 0.006 | 0 | 0 | 8 |
| `layer_0.dynamic_b_projection` | 1.188 | 0.002 | 191 | 32 | 0 |
| `layer_0.dynamic_c_projection` | 1.182 | 0.002 | 191 | 32 | 0 |
| `layer_0.dynamic_b_broadcast` | 1.046 | 0.002 | 191 | 16 | 0 |
| `layer_0.dynamic_c_broadcast` | 1.029 | 0.002 | 191 | 16 | 0 |
