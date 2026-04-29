"""
MorphSpec — Extended E2E Figure (v2)
Replaces: old Figure 9

Changes from v1:
  - Each family has its own request rate range:
      LLaMA-3-70B:   0.1 - 0.9 req/s   (large dense model, low absolute throughput)
      Gemma-3-27B:   1.0 - 7.0 req/s   (mid-size, higher throughput ceiling)
      Qwen3-235B:    3.0 - 10.0 req/s  (MoE, sparse activation -> high effective throughput)
  - SGLang-L3 replaced with vLLM-L2
  - Trends are diversified per (family x workload):
      * ShareGPT (short in/out):   MorphSpec keeps ahead through most of the range
      * HumanEval (mid-length):    L3 collapses earlier; MorphSpec shifts down cleanly
      * GSM8K (long reasoning):    MorphSpec shines on tail latency; throughput stays flat longer
  - MorphSpec is always on the upper envelope, but *how* it wins differs per panel.

INSTRUCTIONS: replace the numbers in `LLAMA3`, `GEMMA3`, `QWEN235` with your real measurements.
Every entry: (rate_array, norm_throughput_array, norm_p90_latency_array).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# =============================================================================
# Style
# =============================================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLOR = {
    "morphspec":  "#c0392b",
    "static_l3":  "#27ae60",
    "static_l2":  "#e67e22",
    "vllm_l2":    "#8e44ad",
    "ar":         "#2c3e50",
}
MARKER = {
    "morphspec":  "o",
    "static_l3":  "s",
    "static_l2":  "^",
    "vllm_l2":    "v",
    "ar":         "D",
}
LABEL = {
    "morphspec":  "MorphSpec (Ours)",
    "static_l3":  "SGLang-L3",
    "static_l2":  "SGLang-L2",
    "vllm_l2":    "vLLM-L2",
    "ar":         "SGLang (AR)",
}

# =============================================================================
# DATA
# =============================================================================

# ---- LLaMA-3-70B: rates 0.1 - 0.9 ----
R_LLAMA = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])

LLAMA3 = {
    # ShareGPT: short prompts, MorphSpec stays near L3 most of the range then bails out late
    "ShareGPT": {
        "morphspec": (R_LLAMA, [1.78, 1.81, 1.75, 1.55, 1.20, 1.04],
                               [1.05, 1.10, 1.18, 1.32, 1.44, 1.52]),
        "static_l3": (R_LLAMA, [1.76, 1.79, 1.70, 1.32, 0.72, 0.45],
                               [1.08, 1.16, 1.30, 1.72, 2.35, 2.62]),
        "static_l2": (R_LLAMA, [1.48, 1.50, 1.47, 1.32, 1.05, 0.90],
                               [1.06, 1.13, 1.22, 1.42, 1.75, 2.10]),
        "vllm_l2":   (R_LLAMA, [1.38, 1.40, 1.35, 1.18, 0.95, 0.82],
                               [1.10, 1.18, 1.30, 1.55, 1.90, 2.25]),
        "ar":        (R_LLAMA, [1.00]*6, [1.00]*6),
    },
    # HumanEval: L3 collapses EARLIER (around 0.5); MorphSpec transitions smoothly
    "HumanEval": {
        "morphspec": (R_LLAMA, [1.72, 1.74, 1.62, 1.38, 1.15, 1.02],
                               [1.08, 1.14, 1.22, 1.38, 1.52, 1.65]),
        "static_l3": (R_LLAMA, [1.70, 1.72, 1.55, 0.92, 0.48, 0.38],
                               [1.12, 1.24, 1.48, 2.10, 2.60, 2.80]),
        "static_l2": (R_LLAMA, [1.52, 1.55, 1.50, 1.35, 1.08, 0.88],
                               [1.08, 1.15, 1.26, 1.48, 1.80, 2.18]),
        "vllm_l2":   (R_LLAMA, [1.40, 1.42, 1.36, 1.20, 0.98, 0.78],
                               [1.12, 1.20, 1.34, 1.60, 1.98, 2.40]),
        "ar":        (R_LLAMA, [1.00]*6, [1.00]*6),
    },
    # GSM8K: long reasoning; all systems degrade slower; MorphSpec wins on latency stability
    "GSM8K": {
        "morphspec": (R_LLAMA, [1.82, 1.80, 1.76, 1.62, 1.30, 1.08],
                               [1.03, 1.07, 1.13, 1.25, 1.38, 1.48]),
        "static_l3": (R_LLAMA, [1.80, 1.78, 1.72, 1.48, 0.95, 0.52],
                               [1.05, 1.12, 1.25, 1.58, 2.12, 2.55]),
        "static_l2": (R_LLAMA, [1.44, 1.45, 1.42, 1.35, 1.18, 1.00],
                               [1.04, 1.09, 1.16, 1.30, 1.55, 1.88]),
        "vllm_l2":   (R_LLAMA, [1.34, 1.36, 1.33, 1.24, 1.06, 0.88],
                               [1.08, 1.15, 1.24, 1.42, 1.72, 2.08]),
        "ar":        (R_LLAMA, [1.00]*6, [1.00]*6),
    },
}

# ---- Gemma-3-27B: rates 1 - 7, L2-only, peak ~1.41x ----
R_GEMMA = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

GEMMA3 = {
    # ShareGPT: MorphSpec holds L3 then steps down; L3 collapses by r=4
    "ShareGPT": {
        "morphspec": (R_GEMMA, [1.41, 1.40, 1.38, 1.32, 1.18, 1.05, 1.00],
                               [1.04, 1.10, 1.18, 1.28, 1.38, 1.46, 1.52]),
        "static_l3": (R_GEMMA, [1.40, 1.38, 1.30, 1.05, 0.75, 0.52, 0.40],
                               [1.08, 1.20, 1.38, 1.75, 2.20, 2.58, 2.90]),
        "static_l2": (R_GEMMA, [1.32, 1.30, 1.26, 1.15, 0.95, 0.75, 0.62],
                               [1.08, 1.18, 1.32, 1.55, 1.85, 2.15, 2.40]),
        "vllm_l2":   (R_GEMMA, [1.22, 1.20, 1.14, 1.02, 0.85, 0.68, 0.56],
                               [1.12, 1.22, 1.38, 1.65, 1.98, 2.30, 2.58]),
        "ar":        (R_GEMMA, [1.00]*7, [1.00]*7),
    },
    # HumanEval: L3 drops earlier (mid-length prompts), MorphSpec transitions at r=3
    "HumanEval": {
        "morphspec": (R_GEMMA, [1.35, 1.34, 1.30, 1.18, 1.04, 1.00, 1.00],
                               [1.06, 1.12, 1.22, 1.36, 1.46, 1.54, 1.60]),
        "static_l3": (R_GEMMA, [1.34, 1.30, 1.18, 0.85, 0.58, 0.42, 0.34],
                               [1.10, 1.25, 1.52, 2.00, 2.45, 2.78, 3.00]),
        "static_l2": (R_GEMMA, [1.26, 1.24, 1.18, 1.05, 0.82, 0.65, 0.52],
                               [1.10, 1.22, 1.42, 1.70, 2.05, 2.38, 2.65]),
        "vllm_l2":   (R_GEMMA, [1.16, 1.14, 1.06, 0.92, 0.74, 0.58, 0.48],
                               [1.14, 1.28, 1.48, 1.80, 2.18, 2.52, 2.80]),
        "ar":        (R_GEMMA, [1.00]*7, [1.00]*7),
    },
    # GSM8K: long decode -> L3 benefit persists slightly longer; MorphSpec stays at L3 then L2
    "GSM8K": {
        "morphspec": (R_GEMMA, [1.40, 1.40, 1.38, 1.33, 1.22, 1.10, 1.02],
                               [1.04, 1.08, 1.14, 1.22, 1.32, 1.42, 1.50]),
        "static_l3": (R_GEMMA, [1.38, 1.37, 1.32, 1.15, 0.82, 0.58, 0.45],
                               [1.06, 1.16, 1.32, 1.65, 2.08, 2.48, 2.80]),
        "static_l2": (R_GEMMA, [1.30, 1.30, 1.27, 1.20, 1.05, 0.85, 0.70],
                               [1.06, 1.14, 1.24, 1.42, 1.68, 1.95, 2.22]),
        "vllm_l2":   (R_GEMMA, [1.18, 1.16, 1.12, 1.04, 0.90, 0.74, 0.62],
                               [1.10, 1.20, 1.32, 1.52, 1.80, 2.10, 2.40]),
        "ar":        (R_GEMMA, [1.00]*7, [1.00]*7),
    },
}

# ---- Qwen3-235B MoE: rates 3 - 10 ----
R_QWEN = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

QWEN235 = {
    # ShareGPT: MorphSpec edges slightly ahead; L3 collapses hard on MoE
    "ShareGPT": {
        "morphspec": (R_QWEN, [1.09, 1.08, 1.06, 1.04, 1.02, 1.00, 1.00, 1.00],
                              [1.02, 1.05, 1.10, 1.16, 1.22, 1.28, 1.34, 1.40]),
        "static_l3": (R_QWEN, [1.06, 1.02, 0.92, 0.78, 0.62, 0.48, 0.38, 0.32],
                              [1.08, 1.18, 1.35, 1.60, 1.92, 2.28, 2.60, 2.88]),
        "static_l2": (R_QWEN, [1.05, 1.03, 0.98, 0.88, 0.72, 0.58, 0.48, 0.40],
                              [1.05, 1.12, 1.24, 1.42, 1.68, 1.98, 2.25, 2.52]),
        "vllm_l2":   (R_QWEN, [0.96, 0.92, 0.85, 0.75, 0.62, 0.50, 0.42, 0.35],
                              [1.10, 1.20, 1.35, 1.55, 1.82, 2.12, 2.42, 2.70]),
        "ar":        (R_QWEN, [1.00]*8, [1.00]*8),
    },
    # HumanEval: even flatter for MorphSpec; L3 drops steeply
    "HumanEval": {
        "morphspec": (R_QWEN, [1.07, 1.06, 1.05, 1.03, 1.01, 1.00, 1.00, 1.00],
                              [1.03, 1.07, 1.12, 1.18, 1.24, 1.30, 1.36, 1.42]),
        "static_l3": (R_QWEN, [1.05, 0.98, 0.85, 0.70, 0.55, 0.42, 0.34, 0.28],
                              [1.10, 1.22, 1.42, 1.70, 2.05, 2.40, 2.72, 3.00]),
        "static_l2": (R_QWEN, [1.03, 1.00, 0.93, 0.82, 0.68, 0.54, 0.44, 0.36],
                              [1.06, 1.15, 1.28, 1.48, 1.75, 2.05, 2.35, 2.62]),
        "vllm_l2":   (R_QWEN, [0.93, 0.88, 0.80, 0.70, 0.58, 0.46, 0.38, 0.32],
                              [1.12, 1.25, 1.42, 1.62, 1.90, 2.20, 2.52, 2.82]),
        "ar":        (R_QWEN, [1.00]*8, [1.00]*8),
    },
    # GSM8K: longer outputs help MoE decode; MorphSpec keeps small L2 edge longer
    "GSM8K": {
        "morphspec": (R_QWEN, [1.08, 1.08, 1.07, 1.05, 1.04, 1.02, 1.00, 1.00],
                              [1.02, 1.04, 1.08, 1.14, 1.20, 1.26, 1.32, 1.38]),
        "static_l3": (R_QWEN, [1.06, 1.04, 0.95, 0.82, 0.68, 0.54, 0.44, 0.36],
                              [1.06, 1.14, 1.30, 1.52, 1.82, 2.18, 2.50, 2.78]),
        "static_l2": (R_QWEN, [1.05, 1.04, 1.00, 0.92, 0.78, 0.64, 0.54, 0.46],
                              [1.04, 1.10, 1.20, 1.36, 1.60, 1.88, 2.15, 2.42]),
        "vllm_l2":   (R_QWEN, [0.95, 0.92, 0.86, 0.76, 0.64, 0.52, 0.44, 0.38],
                              [1.08, 1.18, 1.32, 1.52, 1.78, 2.05, 2.35, 2.62]),
        "ar":        (R_QWEN, [1.00]*8, [1.00]*8),
    },
}

# =============================================================================
# Plotting
# =============================================================================

FAMILIES = [
    {
        "name":       "LLaMA-3-70B Family",
        "data":       LLAMA3,
        "strategies": ["morphspec", "static_l3", "static_l2", "vllm_l2", "ar"],
        "xticks":     [0.1, 0.5, 0.9],
        "xlim":       (0.05, 0.95),
    },
    {
        "name":       "Gemma-3-27B Family",
        "data":       GEMMA3,
        "strategies": ["morphspec", "static_l3", "static_l2", "vllm_l2", "ar"],
        "xticks":     [1, 3, 5, 7],
        "xlim":       (0.7, 7.3),
    },
    {
        "name":       "Qwen3-235B-A22B MoE Family",
        "data":       QWEN235,
        "strategies": ["morphspec", "static_l3", "static_l2", "vllm_l2", "ar"],
        "xticks":     [3, 5, 7, 10],
        "xlim":       (2.5, 10.5),
    },
]
WORKLOADS = ["ShareGPT", "HumanEval", "GSM8K"]

fig, axes = plt.subplots(
    nrows=2, ncols=9,
    figsize=(15, 4.3),
    gridspec_kw={"wspace": 0.38, "hspace": 0.32,
                 "left": 0.048, "right": 0.995,
                 "top": 0.86, "bottom": 0.14}
)

col_idx = 0
for fam_i, fam in enumerate(FAMILIES):
    for wl_j, wl in enumerate(WORKLOADS):

        # ---- TOP: Throughput ----
        ax_t = axes[0, col_idx]
        for strat in fam["strategies"]:
            x, tp, _ = fam["data"][wl][strat]
            ax_t.plot(x, tp,
                      marker=MARKER[strat], color=COLOR[strat],
                      mfc="white" if strat != "morphspec" else COLOR[strat],
                      mew=1.1, linewidth=1.2 if strat != "morphspec" else 1.5,
                      zorder=5 if strat == "morphspec" else 3)
        ax_t.axhline(1.0, color="grey", linestyle=":", linewidth=0.6, zorder=0)
        ax_t.set_ylim(0.25, 2.1)
        ax_t.set_yticks([0.5, 1.0, 1.5, 2.0])
        ax_t.grid(True, axis="y", alpha=0.25, linewidth=0.4)
        ax_t.set_xticks(fam["xticks"])
        ax_t.set_xticklabels([])
        ax_t.set_xlim(fam["xlim"])
        if col_idx == 0:
            ax_t.set_ylabel("Norm. Throughput", fontsize=9)
        if wl_j == 1:
            ax_t.set_title(fam["name"], fontsize=10, pad=10,
                           fontweight="normal", fontstyle="italic")
        ax_t.text(0.5, 0.94, wl, transform=ax_t.transAxes,
                  ha="center", va="top", fontsize=8.5, fontfamily="serif")

        # ---- BOTTOM: P90 latency ----
        ax_l = axes[1, col_idx]
        for strat in fam["strategies"]:
            x, _, lat = fam["data"][wl][strat]
            ax_l.plot(x, lat,
                      marker=MARKER[strat], color=COLOR[strat],
                      mfc="white" if strat != "morphspec" else COLOR[strat],
                      mew=1.1, linewidth=1.2 if strat != "morphspec" else 1.5,
                      zorder=5 if strat == "morphspec" else 3)
        ax_l.axhline(1.0, color="grey", linestyle=":", linewidth=0.6, zorder=0)
        ax_l.set_ylim(0.9, 3.1)
        ax_l.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
        ax_l.grid(True, axis="y", alpha=0.25, linewidth=0.4)
        ax_l.set_xticks(fam["xticks"])
        ax_l.set_xlim(fam["xlim"])
        ax_l.set_xlabel("Request Rate (req/s)", fontsize=8.5)
        if col_idx == 0:
            ax_l.set_ylabel("Norm. P90 Latency", fontsize=9)

        col_idx += 1

# Family divider lines
for divider_x in [0.357, 0.668]:
    line = plt.Line2D([divider_x, divider_x], [0.07, 0.88],
                      transform=fig.transFigure,
                      color="grey", linewidth=0.4,
                      linestyle="-", alpha=0.35)
    fig.add_artist(line)

# Shared legend
handles = [
    Line2D([0], [0], marker="o", color=COLOR["morphspec"], mfc=COLOR["morphspec"],
           mew=1.1, linewidth=1.5, label=LABEL["morphspec"]),
    Line2D([0], [0], marker="s", color=COLOR["static_l3"], mfc="white",
           mew=1.1, linewidth=1.2, label=LABEL["static_l3"]),
    Line2D([0], [0], marker="^", color=COLOR["static_l2"], mfc="white",
           mew=1.1, linewidth=1.2, label=LABEL["static_l2"]),
    Line2D([0], [0], marker="v", color=COLOR["vllm_l2"], mfc="white",
           mew=1.1, linewidth=1.2, label=LABEL["vllm_l2"]),
    Line2D([0], [0], marker="D", color=COLOR["ar"], mfc="white",
           mew=1.1, linewidth=1.2, label=LABEL["ar"]),
]
fig.legend(handles=handles, loc="upper center",
           bbox_to_anchor=(0.5, 1.00), ncol=5, frameon=False,
           columnspacing=2.3, handletextpad=0.5, handlelength=2.2)

plt.savefig("fig_e2e_extended.pdf", bbox_inches="tight", dpi=300)
plt.savefig("fig_e2e_extended.png", bbox_inches="tight", dpi=200)
print("Saved: fig_e2e_extended.pdf / .png")
