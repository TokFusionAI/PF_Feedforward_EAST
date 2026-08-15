"""生成 configs_ablation/ 下 9 个正式 + 9 个 smoke 配置. 只需 PyYAML, 登录节点可跑.

用法:
    python scripts_ablation/_gen_configs.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs_ablation"
SMOKE = OUT / "smoke"
DATASET_ROOT = "/data/PF_ATIME_dataset"

# (config_name, model_name, pe_mode, causal_or_None, extra_model_keys)
CONFIGS = [
    ("transformer_on", "transformer", "time", True, {"n_layers": 6, "n_heads": 8, "d_ff": 1024}),
    ("transformer_off", "transformer", "none", True, {"n_layers": 6, "n_heads": 8, "d_ff": 1024}),
    ("transformer_bidir_on", "transformer", "time", False, {"n_layers": 6, "n_heads": 8, "d_ff": 1024}),
    ("transformer_bidir_off", "transformer", "none", False, {"n_layers": 6, "n_heads": 8, "d_ff": 1024}),
    ("transformer_bidir_learnpe", "transformer", "learnable", False, {"n_layers": 6, "n_heads": 8, "d_ff": 1024}),
    ("lstm_on", "lstm", "time", None, {"n_layers": 2}),
    ("lstm_off", "lstm", "none", None, {"n_layers": 2}),
    ("mlp_on", "mlp", "time", None, {"hidden_dims": [512, 256]}),
    ("mlp_off", "mlp", "none", None, {"hidden_dims": [512, 256]}),
]


def build(cfg_name, name, pe_mode, causal, extra, smoke=False):
    extra = dict(extra)
    model = {
        "name": name,
        "pe_mode": pe_mode,
        "d_state": 19,
        "d_action": 12,
        "d_model": 64 if smoke else 256,
        "dropout": 0.1,
        "T_max": 128,
        "pe_base_period_s": 20.0,
    }
    if causal is not None:
        model["causal"] = causal
    if smoke:
        if "n_layers" in extra:
            extra["n_layers"] = 2
        if "n_heads" in extra:
            extra["n_heads"] = 4
        if "d_ff" in extra:
            extra["d_ff"] = 256
    model.update(extra)

    data = {
        "dataset_root": DATASET_ROOT,
        "train_shots": "meta/train_shots.txt",
        "val_shots": "meta/val_shots.txt",
        "test_shots": "meta/test_shots.txt",
        "norm_stats": "meta/norm_stats_notime.npz",
        "T_max": 128,
        "num_workers": 8 if smoke else 6,  # 8 rank × 6 = 48 worker + 8 主进程 ≈ 56, 留余量给 64 CPU
        "pin_memory": True,
        "prefetch_factor": 4,
        "persistent_workers": True,
    }
    if smoke:
        data["train_subset_n"] = 1000
        data["val_subset_n"] = 200

    return {
        "exp_name": (cfg_name + "_smoke") if smoke else cfg_name,
        "seed": 20260424,
        "data": data,
        "model": model,
        "optim": {
            "lr": 3.0e-4,
            "weight_decay": 1.0e-2,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "amp_dtype": "bf16",
            "epochs": 3 if smoke else 60,
            "batch_per_rank": 16 if smoke else 32,
            "scheduler": "onecycle",
            "warmup_pct": 0.1 if smoke else 0.05,
        },
        "log": {
            "out_root": f"results/ablation/{cfg_name}",
            "tb_dir": f"results/ablation/{cfg_name}/tb",
            "log_every": 20 if smoke else 50,
            "save_every_epoch": 1 if smoke else 5,
        },
        "ddp": {"backend": "nccl", "find_unused_parameters": False},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SMOKE.mkdir(parents=True, exist_ok=True)
    for cfg_name, name, pe_mode, causal, extra in CONFIGS:
        for smoke, dir_ in [(False, OUT), (True, SMOKE)]:
            d = build(cfg_name, name, pe_mode, causal, extra, smoke=smoke)
            suffix = "_smoke" if smoke else ""
            path = dir_ / f"ablation_{cfg_name}{suffix}.yaml"
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(d, f, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100)
            print(f"wrote {path.relative_to(ROOT)}  (model.name={name} pe_mode={pe_mode} causal={causal})")
    print(f"\n{len(CONFIGS)} full + {len(CONFIGS)} smoke configs written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
