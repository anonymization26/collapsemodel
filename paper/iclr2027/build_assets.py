"""Generate manuscript tables/figures from committed outcomes, without fitting models."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/collapse-iclr-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E2B = ROOT / "results/target_conditioned/e2b_fixed_cost_shortlist/domainnet_v1"
E2A = ROOT / "results/target_conditioned/e2_dataset_selection/oracle_headroom_v1"
INPUTS: dict[str, str] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def tex_escape(value: str) -> str:
    escapes = {"%": r"\%", "&": r"\&", "_": r"\_", "#": r"\#"}
    if any(c in value for c in "\\{}$~^"):
        raise ValueError("Abstract must be plain text without TeX control syntax")
    return "".join(escapes.get(c, c) for c in value)


def main() -> None:
    generated, figures = HERE / "generated", HERE / "figures"
    generated.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    summary_path = E2B / "summary/summary.json"
    summary = json.loads(summary_path.read_text())
    INPUTS[str(summary_path.relative_to(ROOT))] = digest(summary_path)
    table_path = E2B / "summary/method_summary.csv"
    if digest(table_path) != summary["method_summary_file_sha256"]:
        raise ValueError("E2b summary table hash mismatch")
    table = rows(table_path)
    index = {(r["method"], int(r["shortlist_size"])): r for r in table}
    assert len(index) == len(table) == 40
    a, m = index["target_a", 227], index["second_moment_mmd", 227]
    for field in ("mean_true_top_q_recall", "mean_selected_normalized_regret"):
        assert float(a[field]) == float(m[field])
    assert summary["same_information_selection_comparison"]["matching_validation_selected_combinations"] == 12
    labels = {
        "target_a": "Target A-opt", "second_moment_mmd": "Second-moment MMD",
        "bayesian_d": "Bayesian D-opt", "merged_effective_rank": "Merged effective rank",
        "random": "Random (20 repeats)", "target_energy": "Target energy",
        "dpp_subspace": "DPP subspace", "domain_balance": "Domain balance",
    }
    output = []
    for key, label in labels.items():
        r = index[key, 227]
        output.append(f"{label} & {100 * float(r['mean_true_top_q_recall']):.3f} & "
                      f"{100 * float(r['mean_selected_normalized_regret']):.6f} & "
                      f"{float(r['mean_screen_spearman_vs_test_brier']):.4f} " + r"\\")
    (generated / "primary_rows.tex").write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        + "Screen & Recall (\\%) & Regret (\\%) & Spearman\\\\\n\\midrule\n"
        + "\n".join(output) + "\n\\bottomrule\n\\end{tabular}\n")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    sizes = [10, 25, 50, 100, 227]
    fig, ax = plt.subplots(figsize=(6.1, 2.8), layout="constrained")
    colors = {"target_a": "#167D9A", "second_moment_mmd": "#B63855",
              "merged_effective_rank": "#667D24", "random": "#555555"}
    for key, color in colors.items():
        values = [index[key, s] for s in sizes]
        ax.plot(sizes, [100 * float(r["mean_true_top_q_recall"]) for r in values],
                marker="o", ms=3, label=labels[key], color=color)
    ax.axhline(90, color="#9A9A9A", ls=":", lw=1)
    ax.set(xlabel="Shortlist size (455 equal-cost combinations)",
           ylabel="Independent-test top-10 recall (%)", ylim=(0, 103), xticks=sizes)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.15)
    fig.savefig(figures / "shortlist_recall.pdf", metadata={"CreationDate": None})
    plt.close(fig)

    headroom = rows(E2A / "summary/primary_headroom_units.csv")
    spans = []
    for encoder in ("resnet50", "dinov2_b14"):
        combinations = rows(E2B / f"test_audit/{encoder}/combinations.csv")
        for domain in sorted({r["target_domain"] for r in combinations}):
            task = [r for r in combinations if r["target_domain"] == domain]
            assert len(task) == len({r["combination"] for r in task}) == 455
            assert all(int(r["selected_sample_count"]) == 1536 for r in task)
            brier = np.array([float(r["brier_score"]) for r in task])
            span = 100 * float((brier.max() - brier.min()) / brier.min())
            spans.append({"encoder": encoder, "domain": domain,
                          "relative_brier_span_percent": span})
    assert max(r["relative_brier_span_percent"] for r in spans) < .425
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 3.0), layout="constrained")
    names = [("OH " if r["dataset"] == "office_home" else "PACS ") +
             r["target_domain"].replace("art_painting", "painting").replace("Real World", "Real")
             for r in headroom]
    axes[0].barh(names, [100 * float(r["oracle_relative_headroom_from_baseline"])
                        for r in headroom], color="#167D9A")
    axes[0].axvline(2, ls=":", color="#B63855", label="2% effect threshold")
    axes[0].set(xlabel="Oracle headroom over DPP (%)", title="PACS / Office-Home (post hoc)")
    axes[0].tick_params(axis="y", labelsize=7)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    domains = sorted({r["domain"] for r in spans})
    y = np.arange(len(domains))
    for i, (enc, color) in enumerate((("resnet50", "#167D9A"), ("dinov2_b14", "#B63855"))):
        vals = [next(r["relative_brier_span_percent"] for r in spans
                     if r["encoder"] == enc and r["domain"] == d) for d in domains]
        axes[1].barh(y + (i - .5) * .34, vals, height=.32, color=color, label=enc)
    axes[1].set(yticks=y, yticklabels=domains, xlabel="Full relative Brier span (%)",
                title="DomainNet (post hoc)", xlim=(0, .45))
    axes[1].tick_params(axis="y", labelsize=7)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=7, frameon=False,
               loc="outside lower center", ncol=2)
    fig.savefig(figures / "headroom_diagnostics.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    (generated / "numbers.tex").write_text(
        "% Generated from committed E2b outcome tables.\n"
        + f"\\newcommand{{\\MainRecall}}{{{100 * float(a['mean_true_top_q_recall']):.3f}}}\n"
        + f"\\newcommand{{\\MainRegret}}{{{100 * float(a['mean_selected_normalized_regret']):.6f}}}\n"
        + f"\\newcommand{{\\MainReduction}}{{{100 * float(a['mean_shortlist_reduction']):.3f}}}\n")
    abstracts = {}
    for lang in ("en", "zh"):
        path = HERE / f"abstract_{lang}.txt"
        value = " ".join(path.read_text(encoding="utf-8").split())
        INPUTS[str(path.relative_to(ROOT))] = digest(path)
        abstracts[lang] = value
        (generated / f"abstract_{lang}.tex").write_text(tex_escape(value) + "\n", encoding="utf-8")
    (HERE / "abstract.md").write_text(
        "# ICLR 2027 Draft Abstract\n\n"
        "Generated by `build_assets.py`; the paper inputs the same text.\n\n"
        "## Plain-text fallback (no math rendering required)\n\n" + abstracts["en"]
        + "\n\n## 中文对照\n\n" + abstracts["zh"] + "\n", encoding="utf-8")
    report = {"input_sha256": INPUTS, "e2b_summary_id": summary["summary_id"],
              "diagnostic_status": "post-hoc description; no changed gates or new experiments",
              "domainnet_brier_spans": spans,
              "primary_metrics": summary["primary_result"]}
    (generated / "asset_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("Verified 40 summary rows, 12 complete 455-combination tasks, and equal primary A-opt/MMD metrics.")


if __name__ == "__main__":
    main()
