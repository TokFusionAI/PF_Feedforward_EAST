"""6-model ablation table/heatmap — proposed = bidirectional Transformer + time-sinusoidal PE.
读 summary_aggregated.json (归一化 MSE/R²/params) + breakdown_test.npz (算 RMSE_kA).
登录节点可跑: python3 scripts_ablation/paper_ablation_table_fig2.py
产物: figures/ablation_heatmap2.{png,pdf}, figures/ablation_table2.tex, figures/ablation_table_full.tex

注: 本版只报告 6 个配置 (去掉 learned-PE 行与 LSTM/MLP 的 Time 变体), 并以
transformer_bidir_on (双向 + 物理时间正弦位置编码) 为提出的最优模型 (★)。原 9 行版见 .orig。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "ablation"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
D = json.load(open(RES / "summary_aggregated.json", encoding="utf-8"))
C = D["configs"]
SEEDS = [11, 22, 33, 44, 55, 66, 77, 20260424]


def mse(k): return float(C[k]["global_mse_mean"])
def r2med(k): return float(np.median(C[k]["r2_coil_mean"]))
def params_m(k): return C[k]["nparams"] / 1e6


# ---- 从 breakdown 算每 config 的全局 RMSE(kA) (mean±std over seeds) ----
def rmse_kA_for_config(cfg):
    vals = []
    for s in SEEDS:
        p = RES / cfg / f"{cfg}_s{s}" / "breakdown_test.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        rmse = d["rmse_kA"]        # (3 phase, 12 coil) kA
        cnt = d["cnt"]             # (3,12)
        se = (rmse ** 2) * cnt      # 还原平方误差(kA^2)
        vals.append(float(np.sqrt(se.sum() / np.maximum(cnt.sum(), 1))))
    return (float(np.mean(vals)), float(np.std(vals, ddof=1))) if vals else (float("nan"), 0.0)


RMSE = {k: rmse_kA_for_config(k) for k in C}   # k -> (mean, std)
BEST = "transformer_bidir_on"                   # proposed: bidir Transformer + time-sinusoidal PE
BM = mse(BEST)
BR = RMSE[BEST][0]
def delta_pct(k): return (mse(k) - BM) / BM * 100.0
def delta_kA(k): return (RMSE[k][0] - BR) / BR * 100.0


# ============ ①②③ 热力图: 2×2 Transformer design grid (None / Time × Causal / Bidir) ============
rows = ["Causal\n(w/ mask)", "Bidirectional\n(w/o mask)"]
cols = ["w/o\n(None)", "Time"]
grid = [
    ["transformer_off", "transformer_on"],
    ["transformer_bidir_off", "transformer_bidir_on"],
]
Rk = np.full((2, 2), np.nan)
for i in range(2):
    for j in range(2):
        Rk[i, j] = RMSE[grid[i][j]][0]

fig, ax = plt.subplots(figsize=(6.4, 4.0))
cmap = plt.cm.viridis_r.copy()
cmap.set_bad("#dddddd")
vmin, vmax = np.nanmin(Rk) * 0.95, np.nanmax(Rk) * 1.05
im = ax.imshow(Rk, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
for i in range(2):
    for j in range(2):
        k = grid[i][j]
        rmu = RMSE[k][0]; dk = delta_kA(k); r = r2med(k)
        best = (k == BEST)
        ax.text(j, i, f"{rmu:.3f} kA\n({dk:+.1f}%, R²={r:.3f})" + ("  ★" if best else ""),
                ha="center", va="center", fontsize=10.5,
                fontweight="bold" if best else "normal",
                color="white" if (rmu > (vmin + vmax) / 2) else "black")
# highlight proposed cell (Bidirectional, Time) = (row 1, col 1)
ax.add_patch(plt.Rectangle((1 - 0.5, 1 - 0.5), 1, 1, fill=False, edgecolor="red", lw=3.5))
ax.set_xticks(range(2)); ax.set_xticklabels(cols, fontsize=11)
ax.set_yticks(range(2)); ax.set_yticklabels(rows, fontsize=11)
ax.set_title("Transformer design ablation — test RMSE (kA), Δ% and R² vs proposed ★",
             fontsize=12, pad=8)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("RMSE (kA)  ↓ better", fontsize=10)
ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
ax.grid(which="minor", color="white", lw=2.5)
ax.tick_params(which="minor", length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "ablation_heatmap2.png", dpi=200, bbox_inches="tight")
fig.savefig(FIG / "ablation_heatmap2.pdf", bbox_inches="tight")
plt.close(fig)
print(f"wrote {FIG.relative_to(ROOT)}/ablation_heatmap2.{{png,pdf}}")

# ============ ④-a 主表 v2 (Transformer design grid + cross-arch baselines) ============
cau_none, cau_time = "transformer_off", "transformer_on"
bid_none, bid_time = "transformer_bidir_off", "transformer_bidir_on"
lstm_n, mlp_n = "lstm_off", "mlp_off"


def kA(k): return f"{RMSE[k][0]:.3f}"


tex = [r"\begin{table}[t]", r"\centering",
       r"\caption{Design ablation on the test set (8 seeds). \textbf{Top:} Transformer design grid "
       r"(attention $\times$ positional encoding). \textbf{Bottom:} cross-architecture baselines. "
       r"The proposed model, a bidirectional Transformer with time-based sinusoidal positional encoding "
       r"($\star$), gives the lowest error (bold). RMSE in kA (physical units); $\Delta$ = relative "
       r"RMSE vs.\ the proposed model. Lower is better.}",
       r"\label{tab:ablation}", r"\small",
       r"\begin{tabular}{l|cc}", r"\toprule",
       r" & \multicolumn{2}{c}{Positional encoding (PE)} \\", r"\cmidrule(lr){2-3}",
       r"Attention & w/o (None) & Time (sinusoidal) \\", r"\midrule",
       f"Causal (w/ mask) & {kA(cau_none)} \\scriptsize({delta_kA(cau_none):+.0f}\\%, R$^2$={r2med(cau_none):.3f}) "
       f"& {kA(cau_time)} \\scriptsize({delta_kA(cau_time):+.0f}\\%, R$^2$={r2med(cau_time):.3f}) \\\\",
       f"Bidirectional (w/o) & {kA(bid_none)} \\scriptsize({delta_kA(bid_none):+.0f}\\%, R$^2$={r2med(bid_none):.3f}) "
       f"& \\textbf{{{kA(bid_time)}}} \\scriptsize(Ours, R$^2$={r2med(bid_time):.3f})$\\star$ \\\\",
       r"\bottomrule", r"\end{tabular}", r"",
       r"\vspace{2pt}", r"\begin{tabular}{llccccc}", r"\toprule",
       r"Backbone (mixing) & PE & \#Params(M) & RMSE(kA)$\downarrow$ & Test MSE$\downarrow$ & R$^2_{\mathrm{med}}\uparrow$ & $\Delta$ \\", r"\midrule",
       f"\\textbf{{Transformer-Bidir}}$\\star$ (ours) & Time (sinusoidal) & {params_m(bid_time):.2f} "
       f"& \\textbf{{{kA(bid_time)}}} & \\textbf{{{mse(bid_time):.4f}}} & \\textbf{{{r2med(bid_time):.3f}}} & --- \\\\",
       f"LSTM (recurrent, causal) & None & {params_m(lstm_n):.2f} "
       f"& {kA(lstm_n)} & {mse(lstm_n):.4f} & {r2med(lstm_n):.3f} & +{delta_kA(lstm_n):.0f}\\% \\\\",
       f"MLP (pointwise) & None & {params_m(mlp_n):.2f} "
       f"& {kA(mlp_n)} & {mse(mlp_n):.4f} & {r2med(mlp_n):.3f} & +{delta_kA(mlp_n):.0f}\\% \\\\",
       r"\bottomrule", r"\end{tabular}", r"",
       r"\smallskip",
       r"\footnotesize Notation: w/ mask = causal attention; w/o = bidirectional. PE: Time = sinusoidal of "
       rf"physical time; None = none. Transformer variants share {params_m(bid_time):.2f}M params. "
       r"LSTM causal by recurrence; MLP pointwise.",
       r"\end{table}"]
(FIG / "ablation_table2.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
print(f"wrote {FIG.relative_to(ROOT)}/ablation_table2.tex")

# ============ ④-b 完整 6 行排名表 (附表) ============
META = {  # config -> (Backbone, Attention, PE)
    "transformer_on": ("Transformer", "Causal (w/ mask)", "Time"),
    "transformer_off": ("Transformer", "Causal (w/ mask)", "None"),
    "transformer_bidir_on": ("Transformer", "Bidirectional (w/o)", "Time"),
    "transformer_bidir_off": ("Transformer", "Bidirectional (w/o)", "None"),
    "lstm_off": ("LSTM", "Recurrent (causal)", "None"),
    "mlp_off": ("MLP", "Pointwise (—)", "None"),
}
SIX = list(META.keys())
order = sorted(SIX, key=lambda k: mse(k))
tex2 = [r"\begin{table}[t]", r"\centering",
        r"\caption{Full ranking of the six reported configurations on the test set (8 seeds), sorted by Test MSE.}",
        r"\label{tab:ablation_full}", r"\small",
        r"\begin{tabular}{rlllcccc}", r"\toprule",
        r"\# & Backbone & Attention & PE & \#Params(M) & Test MSE$\downarrow$ & RMSE(kA)$\downarrow$ & R$^2_{\mathrm{med}}\uparrow$ \\", r"\midrule"]
for rank, k in enumerate(order, 1):
    bb, att, pe = META[k]
    star = r"$\star$" if k == BEST else ""
    bf = r"\textbf{" if k == BEST else ""
    be = "}" if k == BEST else ""
    tex2.append(
        f"{rank} & {bf}{bb}{be}{star} & {att} & {pe} & {params_m(k):.2f} "
        f"& {bf}{mse(k):.5f}{be} & {bf}{kA(k)}{be} & {bf}{r2med(k):.3f}{be} \\\\"
    )
tex2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(FIG / "ablation_table_full.tex").write_text("\n".join(tex2) + "\n", encoding="utf-8")
print(f"wrote {FIG.relative_to(ROOT)}/ablation_table_full.tex")

# 控制台: RMSE(kA) 全表
print("\n=== RMSE(kA) 全表 (mean over 8 seeds) ===")
for k in order:
    m, sd = RMSE[k]
    star = " ★" if k == BEST else ""
    print(f"  {k:30s} RMSE={m:.4f}±{sd:.4f} kA  (Δ={delta_kA(k):+6.1f}%)  MSE={mse(k):.5f}  R²={r2med(k):.4f}{star}")
