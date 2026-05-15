# Stage 1 Phase Timing Comparison

- Baseline: `runs/stage1-s045-rank-reduce-bc-lora-merged-mamba130m-v03146.json`
- Candidate: `runs/stage1-s045-dt-hidden-reduce-lora-merged-mamba130m-v03146.json`
- Eval speedup: `1.243`

| phase | baseline s | candidate s | delta s | speedup |
|---|---:|---:|---:|---:|
| `layer_0.dt_hidden_projection` | 107.865 | 3.767 | 104.098 | 28.633 |
| `layer_0.conv_projection` | 184.901 | 160.641 | 24.260 | 1.151 |
| `layer_0.gate_projection` | 181.381 | 159.283 | 22.098 | 1.139 |
| `layer_0.output_projection` | 157.599 | 156.480 | 1.119 | 1.007 |
| `layer_0.rank_input_baby_rotations` | 0.041 | 0.000 | 0.041 |  |
| `layer_0.gate_silu_polynomial` | 3.482 | 3.465 | 0.017 | 1.005 |
| `layer_0.dynamic_b_broadcast` | 1.057 | 1.046 | 0.011 | 1.010 |
| `layer_0.dynamic_c_broadcast` | 1.038 | 1.029 | 0.009 | 1.009 |
