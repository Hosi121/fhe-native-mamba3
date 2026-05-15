# Stage 1 Phase Timing Comparison

- Baseline: `runs/stage1-s045-rank-reduce-bc-lora-merged-mamba130m-v03146.json`
- Candidate: `runs/stage1-s045-dt-projection-reduce-lora-merged-mamba130m-v03148.json`
- Eval speedup: `1.526`

| phase | baseline s | candidate s | delta s | speedup |
|---|---:|---:|---:|---:|
| `layer_0.dt_hidden_projection` | 107.865 | 3.213 | 104.652 | 33.573 |
| `layer_0.dt_rank_projection` | 103.416 | 7.320 | 96.096 | 14.129 |
| `layer_0.gate_projection` | 181.381 | 150.216 | 31.165 | 1.207 |
| `layer_0.conv_projection` | 184.901 | 159.319 | 25.582 | 1.161 |
| `layer_0.output_projection` | 157.599 | 155.640 | 1.959 | 1.013 |
| `layer_0.rank_silu_polynomial` | 5.312 | 5.165 | 0.147 | 1.028 |
| `layer_0.gate_silu_polynomial` | 3.482 | 3.366 | 0.116 | 1.034 |
| `layer_0.dynamic_c_projection` | 1.179 | 1.135 | 0.043 | 1.038 |
