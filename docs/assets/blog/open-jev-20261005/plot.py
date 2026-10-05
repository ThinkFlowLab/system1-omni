"""Regenerate the historical Open-Jev figures; CPU only, no model loading."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "source-data.json").read_text())
GRAY, BLUE, TEAL, ORANGE = "#9CA3AF", "#2563EB", "#0F766E", "#D97706"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 11, "svg.fonttype": "none", "svg.hashsalt": "open-jev-20261005",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": "#CBD5E1",
    "text.color": "#172033", "axes.labelcolor": "#334155",
    "xtick.color": "#475569", "ytick.color": "#172033",
})


def bars(ax, rows, title, limit, colors, digits=3, sample_key="pass_mean_ms"):
    for i, (row, color) in enumerate(zip(rows, colors)):
        ax.barh(i, row["mean_ms"], height=0.52, color=color, zorder=2)
        ax.scatter(row[sample_key], [i] * 2, s=23, color="#172033", zorder=3)
        ax.text(row["mean_ms"] + limit * 0.018, i,
                f'{row["mean_ms"]:.{digits}f}', va="center", fontsize=11)
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, limit)
    ax.set_xlabel("Mean wall time (ms) — lower is better")
    ax.set_title(title, loc="left", pad=16, weight="bold")
    ax.grid(axis="x", color="#E2E8F0", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.margins(y=0.3)


def save(fig, name, footer):
    fig.text(0.03, 0.035, footer, fontsize=9, color="#475569")
    svg = HERE / f"{name}.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(HERE / f"{name}.png", dpi=170, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    print(name)


fig, ax = plt.subplots(figsize=(11.6, 4.5))
fig.subplots_adjust(left=0.22, right=0.93, top=0.79, bottom=0.22)
bars(ax, DATA["backend"]["rows"], "Matched full-backend comparison", 415,
     [GRAY, BLUE, TEAL])
fig.suptitle("Open-Jev on H200: 362.21 → 48.50 ms versus raw HF",
             x=0.03, y=0.96, ha="left", fontsize=17, weight="bold")
save(fig, "backend-http",
     "2026-10-03 · BF16 · 74 single-candidate requests/pass · concurrency 1\n"
     "Bars: reported aggregate means. Black dots: two pass means. Historical summary; no confidence intervals.")

fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
fig.subplots_adjust(left=0.13, right=0.96, top=0.75, bottom=0.25, wspace=0.62)
bars(axes[0], DATA["graph"]["mixed"], "74 mixed requests: cache capacity matters", 112,
     [GRAY, ORANGE, BLUE])
bars(axes[1], DATA["graph"]["short"], "Fixed 107-token request: warm replay", 23.5,
     [GRAY, ORANGE, BLUE])
fig.suptitle("CUDA Graph replay: warm gains depend on retaining the workload's lengths",
             x=0.03, y=0.97, ha="left", fontsize=16, weight="bold")
save(fig, "graph-cache",
     "2026-10-03 · H200 · BF16 · concurrency 1 · independent graph-cache experiment\n"
     "Mixed: 74 requests/pass; short: 32 requests/pass. Black dots: two pass means. Different x-axis limits; both start at zero.")

fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
fig.subplots_adjust(left=0.15, right=0.96, top=0.83, bottom=0.18, hspace=0.82, wspace=0.62)
for ax, shape, limit in zip(axes.flat, DATA["gdn_kernel"]["shapes"], [0.063, 0.30, 0.94]):
    bars(ax, shape["rows"], f'{shape["tokens"]:,} tokens · complete GDN call', limit,
         [GRAY, BLUE], digits=5)
bars(axes[1, 1], DATA["gdn_http"]["rows"], "74 requests · warm HTTP", 57,
     [GRAY, BLUE])
fig.suptitle("GDN preparation: ~13% longer-call gains, 0.42% HTTP gain",
             x=0.03, y=0.97, ha="left", fontsize=17, weight="bold")
save(fig, "gdn-kernel-http",
     "2026-10-03 · H200 · two separate A/B experiments · CUDA Graph disabled\n"
     "Kernel: seeded synthetic inputs, 100 calls/pass. HTTP: 74 real requests/pass, concurrency 1.\n"
     "Bars recomputed from raw records; black dots are two pass means. Each panel has its own zero-based scale.")

fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.5))
fig.subplots_adjust(left=0.15, right=0.95, top=0.83, bottom=0.19, hspace=0.88, wspace=0.67)
for row, experiment in enumerate(DATA["isolated_pr55"]):
    label = "RMSNorm · Oct 1 · GPU 5" if row == 0 else "SiLU · Oct 2 · GPU 2"
    bars(axes[row, 0], experiment["http_rows"], f"{label}: warm HTTP", 62,
         [GRAY, BLUE])
    bars(axes[row, 1], experiment["trace_rows"],
         f'{experiment["trace_tokens"]:,} tokens · {experiment["trace_launches"]} launches',
         2.35 if row == 0 else 26, [GRAY, BLUE], sample_key="trace_sum_ms")
    axes[row, 1].set_xlabel("Kernel-family total per trace (ms)")
fig.suptitle("PR #55: isolated RMSNorm and SiLU A/B results",
             x=0.03, y=0.97, ha="left", fontsize=17, weight="bold")
save(fig, "pr55-isolated-ab",
     "H200 · BF16 · graph disabled · each row is an independent campaign with its own paired baseline\n"
     "HTTP: 74 requests/pass, two pass means. Kernels: two separately captured request totals. Black dots show those repetitions.\n"
     "HTTP collection inactive; CUPTI may remain loaded. Panels use different zero-based scales.")

fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.0))
fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.20, hspace=0.85, wspace=0.73)
for row, experiment in enumerate(DATA["regressions"]):
    comparisons = [c for c in experiment["comparisons"] if c["model"] == "jev"]
    for column, comparison in enumerate(comparisons):
        rows = [{"label": arm.capitalize(),
                 "mean_ms": comparison["means"][arm]["mean_latency_ms"],
                 "pass_mean_ms": [p["mean_latency_ms"] for p in comparison["runs"][arm]]}
                for arm in ("baseline", "candidate")]
        bars(axes[row, column], rows,
             f'PR #{experiment["pr"]} · concurrency {comparison["concurrency"]}',
             max(r["mean_ms"] for r in rows) * 1.23, [GRAY, BLUE], digits=2)
        axes[row, column].set_xlabel("Mean HTTP latency (ms)")
fig.suptitle("PRs #78 and #80: Open-Jev regression checks, no speedup claim",
             x=0.03, y=0.97, ha="left", fontsize=16, weight="bold")
save(fig, "pr-regression-ab",
     "2026-10-05 · H200 GPU 5 · BF16 · graph disabled · two independent direct-worker campaigns\n"
     "64 synthetic multi-question/candidate requests/pass; two measured passes per arm and concurrency.\n"
     "Black dots: pass means. Each panel starts at zero and has its own scale. All latency/throughput/p95 regression gates pass.")

fig, axes = plt.subplots(1, 2, figsize=(11.8, 7.4))
fig.subplots_adjust(left=0.16, right=0.92, top=0.79, bottom=0.32, wspace=1.12)
bundle = DATA["backend"]["rows"][:2]
bars(axes[0], bundle, "Complete backend change", 435, [GRAY, BLUE], digits=2)
axes[0].set_xlabel("Mean warm HTTP latency (ms)")
bundle_saved = bundle[0]["mean_ms"] - bundle[1]["mean_ms"]
axes[0].text(0, -0.23,
             f'{bundle_saved:.2f} ms saved · {bundle[0]["mean_ms"] / bundle[1]["mean_ms"]:.2f}×\n'
             "Native kernels + merged LoRA + gate fusion\n"
             "+ cached RMSNorm + packed SiLU\n"
             "Native-only / LoRA-only / gate-only gains: unmeasured",
             transform=axes[0].transAxes, va="top", fontsize=10, color="#475569")

pairs = [("Cached RMSNorm", DATA["isolated_pr55"][0]["http_rows"], GRAY),
         ("Packed BF16 SiLU", DATA["isolated_pr55"][1]["http_rows"], GRAY),
         ("Graph replay: 64 entries", [DATA["graph"]["mixed"][0], DATA["graph"]["mixed"][2]], GRAY),
         ("Cache 8 → 64: recover regression", DATA["graph"]["mixed"][1:], ORANGE)]
ax = axes[1]
for index, (label, rows, baseline_color) in enumerate(pairs):
    for arm, (row, offset, color) in enumerate(zip(rows, [-0.15, 0.15], [baseline_color, BLUE])):
        y = index + offset
        ax.barh(y, row["mean_ms"], height=0.25, color=color, zorder=2,
                label="Baseline" if index == 0 and arm == 0 else "Candidate" if index == 0 else None)
        ax.scatter(row["pass_mean_ms"], [y] * 2, s=16, color="#172033", zorder=3)
        ax.text(row["mean_ms"] + 1.8, y, f'{row["mean_ms"]:.2f}', va="center", fontsize=9)
    saved = rows[0]["mean_ms"] - rows[1]["mean_ms"]
    ax.text(119, index, f'−{saved:.2f} ms\n({saved / rows[0]["mean_ms"] * 100:.2f}%)',
            va="center", fontsize=9, color="#0F766E")
ax.set_yticks(range(len(pairs)), [label for label, _, _ in pairs])
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_xlabel("Mean warm HTTP latency (ms)")
ax.set_title("Isolated A/Bs: separate baselines", loc="left", pad=16, weight="bold")
ax.grid(axis="x", color="#E2E8F0", zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0, pad=12, labelsize=10)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.17), frameon=False, ncol=2, fontsize=9)
fig.suptitle("PR #55: measured gains and missing ablations",
             x=0.03, y=0.96, ha="left", fontsize=18, weight="bold")
save(fig, "pr55-attribution",
     "Historical Oct 1–3 H200 measurements · BF16 · 74 single-candidate requests/pass · two measured passes/arm\n"
     "Right-hand pairs use their own controls: RMSNorm on GPU 5; SiLU/graphs on GPU 2; profiler state differs by campaign.\n"
     "Black dots: pass means. Gray/orange: baseline; blue: candidate. Both axes start at zero and use different scales.\n"
     "These pairs are not consecutive points in a cumulative ladder. LoRA-only, native-only and gate-only stages need fresh A/B runs.")
