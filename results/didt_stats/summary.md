# |dI/dt| phase-aware thresholds

- source: `meta/train_shots.txt`
- shots ok / err: 26638 / 0
- unit: kA/s

## Phase breakdown

| phase | n_shots_with | n_steps | t_total [s] |
|---|---:|---:|---:|
| ramp_up | 26449 | 4,386,108 | 33,531.5 |
| flat_top | 26515 | 24,720,720 | 218,122.1 |
| ramp_down | 22459 | 2,866,452 | 21,895.5 |

## per-channel `max |dI/dt|` by phase (kA/s)

| PCPF | ramp_up | flat_top | ramp_down |
|---|---:|---:|---:|
| PCPF1 | 15.78 | 19.22 | 15.44 |
| PCPF2 | 20.36 | 21.70 | 19.69 |
| PCPF3 | 22.15 | 22.05 | 20.51 |
| PCPF4 | 23.08 | 20.53 | 26.25 |
| PCPF5 | 19.88 | 19.41 | 20.86 |
| PCPF6 | 22.51 | 18.91 | 19.15 |
| PCPF7 | 7.23 | 7.01 | 6.84 |
| PCPF8 | 8.61 | 6.98 | 6.65 |
| PCPF11 | 14.61 | 14.53 | 15.61 |
| PCPF12 | 16.09 | 15.03 | 14.74 |
| PCPF13 | 23.67 | 24.31 | 30.76 |
| PCPF14 | 22.81 | 26.45 | 24.87 |

(也可用 P99.99 或 P99.9 作更稳健的 phase 阈值; 详见 proposed_phase_thresholds.json)