"""Generate manuscript assets from recorded outcomes, without fitting models."""
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
REVISION = ROOT / "results/target_conditioned/revision_20260916"
READOUT = ROOT / "results/target_conditioned/revision_20260917/revision_results/e2b_readout_projection_v1"
BUDGET = ROOT / "results/target_conditioned/revision_20260922/budget_audit"
CHALLENGE = ROOT / "results/target_conditioned/revision_20260922/challenge_summary"
DOMAIN_TIES = ROOT / "results/target_conditioned/revision_20260922/domain_tie_audit"
COST = ROOT / "results/target_conditioned/revision_20260922/cost_summary"
INPUTS: dict[str, str] = {}
METHOD_NAME = "Target A-opt"


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


def supplement_numbers() -> dict[str, float]:
    root = REVISION / "revision_results/e2b_loss_diagnostics_v1"
    loaded = {name: [] for name in ("topq_stability.csv", "shortlist_recall.csv", "metric_agreement.csv")}
    identifiers = []
    for encoder in ("resnet50", "dinov2_b14"):
        path = root / encoder / "manifest.json"
        INPUTS[str(path.relative_to(ROOT))] = digest(path)
        manifest = json.loads(path.read_text())
        core = {k: v for k, v in manifest.items() if k != "diagnostics_id"}
        sealed = hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=True,
                                          separators=(",", ":")).encode()).hexdigest()
        assert sealed == manifest["diagnostics_id"]
        identifiers.append(sealed)
        assert manifest["encoder"] == encoder
        assert manifest["diagnostics_settings"]["bootstrap_repeats"] == 2000
        assert manifest["diagnostics_settings"]["top_q"] == 10
        assert len(manifest["domains"]) == 6
        assert max(d["loss_mean_max_abs_error"] for d in manifest["domains"]) < 1e-10
        for filename in loaded:
            csv_path = root / encoder / filename
            assert digest(csv_path) == manifest["files"][filename]
            loaded[filename].extend(rows(csv_path))
    path = root / "pairing_verification.json"
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    pairing = json.loads(path.read_text())
    assert pairing["status"] == "paired-inputs-verified"
    assert pairing["diagnostics_ids"] == identifiers
    comparison = rows(REVISION / "reconstruction_comparison/per_task.csv")
    assert len(comparison) == 12 and all(float(r["top_q_overlap"]) == 1 for r in comparison)
    definitions = {
        "BootstrapRecall": ("shortlist_recall.csv", "mean", {"metric": "brier_score", "method": "target_a", "shortlist_size": "227"}),
        "BootstrapMMDRecall": ("shortlist_recall.csv", "mean", {"metric": "brier_score", "method": "second_moment_mmd", "shortlist_size": "227"}),
        "BootstrapOverlap": ("topq_stability.csv", "mean", {"metric": "brier_score"}),
        "BrierErrorOverlap": ("metric_agreement.csv", "top_q_overlap", {"left_metric": "brier_score", "right_metric": "error_rate"}),
        "BrierSquaredOverlap": ("metric_agreement.csv", "top_q_overlap", {"left_metric": "brier_score", "right_metric": "squared_loss"}),
        "BrierNLLOverlap": ("metric_agreement.csv", "top_q_overlap", {"left_metric": "brier_score", "right_metric": "nll"}),
    }
    values = {}
    for name, (filename, column, conditions) in definitions.items():
        selected = [r for r in loaded[filename] if all(r[k] == v for k, v in conditions.items())]
        assert len(selected) == len({(r["encoder"], r["target_domain"]) for r in selected}) == 12
        values[name] = 100 * float(np.mean([float(r[column]) for r in selected]))
    return values


def readout_assets(generated: Path) -> dict[str, float]:
    path = READOUT / "summary/summary.json"
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    summary = json.loads(path.read_text())
    for relative, expected in summary["input_sha256"].items():
        source = READOUT / relative
        assert digest(source) == expected, relative
        INPUTS[str(source.relative_to(ROOT))] = expected
    table = rows(READOUT / "summary/new_seeds.csv")
    units = rows(READOUT / "summary/unit_rows.csv")
    key = lambda r: (r["readout"], r["method"], r["metric"], int(r["shortlist_size"]))
    index = {key(r): r for r in table}
    recorded = {key(r): r for r in summary["new_seeds"]}
    assert len(index) == len(table) == len(recorded) == 240
    for k, row in index.items():
        assert int(row["target_domain_count"]) == 6
        assert int(row["repeated_task_count"]) == 48
        assert json.loads(row["projection_seeds"]) == [20260917, 20260918, 20260919, 20260920]
        for field in ("recall_at_top_q", "relative_candidate_span", "relative_regret", "absolute_omission"):
            assert np.isclose(float(row[field]), recorded[k][field], rtol=1e-12, atol=1e-15)
    labels = {
        "target_a": METHOD_NAME, "second_moment_mmd": "Second-moment MMD",
        "bayesian_d": "Bayesian D-opt", "merged_effective_rank": "Merged effective rank",
        "random": "Random (20 repeats)", "target_energy": "Target energy",
        "dpp_subspace": "DPP subspace", "domain_balance": "Domain balance",
    }
    output = []
    for readout, label in (("ridge", "Ridge"), ("logistic", "Logistic")):
        for metric, metric_label in (("brier_score", "Brier"), ("nll", "NLL"), ("error_rate", "Error")):
            values = [100 * float(index[readout, method, metric, 227]["recall_at_top_q"])
                      for method in ("target_a", "second_moment_mmd", "random")]
            output.append(f"{label} & {metric_label} & " + " & ".join(f"{v:.2f}" for v in values) + r"\\")
    (generated / "readout_rows.tex").write_text(
        "\\begin{tabular}{llrrr}\n\\toprule\n"
        + "Readout & Metric & " + METHOD_NAME + " & MMD & Random\\\\\n\\midrule\n"
        + "\n".join(output) + "\n\\bottomrule\n\\end{tabular}\n")
    output = []
    for method, label in labels.items():
        values = [100 * float(index[readout, method, "brier_score", 227]["recall_at_top_q"])
                  for readout in ("ridge", "logistic")]
        output.append(f"{label} & {values[0]:.2f} & {values[1]:.2f} " + r"\\")
    (generated / "readout_all_rows.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\n"
        + "Screen & Ridge & Logistic\\\\\n\\midrule\n"
        + "\n".join(output) + "\n\\bottomrule\n\\end{tabular}\n")
    new_units = [r for r in units if int(r["projection_seed"]) != 20260911
                 and int(r["shortlist_size"]) == 227]
    worst = min((r for r in new_units if r["readout"] == "logistic"
                 and r["method"] == "target_a" and r["metric"] == "error_rate"),
                key=lambda r: float(r["recall_at_top_q"]))
    assert (worst["encoder"], worst["target_domain"], int(worst["projection_seed"])) == (
        "dinov2_b14", "quickdraw", 20260918)
    assert float(worst["absolute_omission"]) == 0
    values = {
        "WorstErrorRecall": 100 * float(worst["recall_at_top_q"]),
        "WorstErrorGap": 100 * float(worst["absolute_regret"]),
    }
    for readout, prefix in (("ridge", "Ridge"), ("logistic", "Logistic")):
        a = index[readout, "target_a", "brier_score", 227]
        m = index[readout, "second_moment_mmd", "brier_score", 227]
        assert float(a["relative_regret"]) == float(m["relative_regret"])
        assert float(a["absolute_omission"]) == float(m["absolute_omission"]) == 0
        values[f"Readout{prefix}A"] = 100 * float(a["recall_at_top_q"])
        values[f"Readout{prefix}MMD"] = 100 * float(m["recall_at_top_q"])
        values[f"{prefix}Span"] = 100 * float(a["relative_candidate_span"])
        values[f"{prefix}SelectedRegret"] = 100 * float(a["relative_regret"])
    assert np.isclose(values["ReadoutRidgeA"], values["ReadoutLogisticA"])
    return values


def budget_assets(generated: Path, figures: Path) -> dict[str, float]:
    curves = rows(BUDGET / "curves.csv")
    failures = rows(BUDGET / "failure_rates.csv")
    rows(BUDGET / "paired_summary.csv")
    path = BUDGET / "summary.json"
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    summary = json.loads(path.read_text())
    for name, expected in summary["input_sha256"].items():
        source = ROOT / name
        assert digest(source) == expected, name
        INPUTS[str(source.relative_to(ROOT))] = expected
    index = {(r["readout"], r["metric"], r["method"], int(r["shortlist_size"])): r for r in curves}
    failure_index = {(r["readout"], r["metric"], r["method"], int(r["shortlist_size"]), float(r["absolute_gap_threshold"])): r for r in failures}
    labels = {"target_a": METHOD_NAME, "second_moment_mmd": "MMD", "random": "Random", "exhaustive": "Exhaustive validation"}
    selected = [("random", 227), ("target_a", 50), ("second_moment_mmd", 50), ("exhaustive", 455)]
    output = []
    for method, size in selected:
        r = index["logistic", "brier_score", method, size]
        failure = failure_index["logistic", "brier_score", method, size, .005]
        output.append(f"{labels[method]} & {size} & {float(r['test_loss']):.6f} & "
                      f"{100 * float(failure['failure_rate']):.2f} " + r"\\")
    (generated / "budget_rows.tex").write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        + "Screen & Fits & Test Brier & Gap $>0.005$ (\\%)\\\\\n\\midrule\n"
        + "\n".join(output) + "\n\\bottomrule\n\\end{tabular}\n")
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.45), layout="constrained")
    sizes = [10, 25, 50, 100, 227]
    for ax, readout, scale, unit in zip(axes, ("ridge", "logistic"), (10000, 1000), ("10,000", "1,000")):
        for method, color in (("target_a", "#167D9A"), ("second_moment_mmd", "#B63855"), ("random", "#555555")):
            ax.plot(sizes, [scale * float(index[readout, "brier_score", method, s]["gap_to_full"]) for s in sizes],
                    marker="o", ms=3, color=color, label=labels[method])
        ax.axhline(0, color="#888888", ls=":", lw=1, label="Exhaustive validation")
        ax.set(title="Ridge-softmax" if readout == "ridge" else "Logistic regression",
               xlabel="Candidates evaluated", ylabel=f"Brier gap x {unit}", xticks=[10, 50, 100, 227])
        ax.grid(axis="y", alpha=.15)
    axes[1].legend(frameon=False, fontsize=6.5)
    fig.savefig(figures / "decision_budget.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    a = index["logistic", "brier_score", "target_a", 50]
    m = index["logistic", "brier_score", "second_moment_mmd", 50]
    return {"BudgetAFifty": float(a["test_loss"]), "BudgetMMDFifty": float(m["test_loss"]),
            "BudgetFull": float(index["logistic", "brier_score", "exhaustive", 455]["test_loss"]),
            "BudgetRandomPrimary": float(index["logistic", "brier_score", "random", 227]["test_loss"]),
            "BudgetFailureFifty": 100 * float(failure_index["logistic", "brier_score", "target_a", 50, .005]["failure_rate"])}


def challenge_assets(generated: Path) -> None:
    for directory in (CHALLENGE, DOMAIN_TIES):
        path = directory / "summary.json"
        INPUTS[str(path.relative_to(ROOT))] = digest(path)
        summary = json.loads(path.read_text())
        assert summary["status"] == "complete"
        for name, expected in summary["input_sha256"].items():
            source = ROOT / name
            assert digest(source) == expected, name
            INPUTS[str(source.relative_to(ROOT))] = expected
    curves = rows(CHALLENGE / "curves.csv") + rows(DOMAIN_TIES / "curves.csv")
    conditional = rows(CHALLENGE / "within_composition.csv")
    index = {(r["construction"], r["readout"], r["metric"], r["method"], int(r["shortlist_size"])): r for r in curves}
    inside = {(r["construction"], r["readout"], r["metric"], r["method"]): r for r in conditional}
    labels = {"target_a": METHOD_NAME, "second_moment_mmd": "MMD",
              "mean_matching": "Mean matching", "domain_moment_ties": "Domain moment (random ties)",
              "random": "Random"}
    output = []
    for method, label in labels.items():
        loss = [float(index[c, "logistic", "brier_score", method, 50]["test_loss"])
                for c in ("original", "hash_partition")]
        if method == "domain_moment_ties":
            # Twenty size-9 and ten size-27 strata per target; uniform tie retention.
            within = [100 * (20 * (4 / 9) + 10 * (13 / 27)) / 30] * 2
        else:
            within = [100 * float(inside[c, "logistic", "brier_score", method]["oracle_retained"])
                      for c in ("original", "hash_partition")]
        output.append(f"{label} & {loss[0]:.6f} & {loss[1]:.6f} & {within[0]:.2f} & {within[1]:.2f} " + r"\\")
    (generated / "construction_rows.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        + " & \\multicolumn{2}{c}{Final Brier ($s=50$)} & \\multicolumn{2}{c}{Within-composition (\\%)}\\\\\n"
        + "Screen & Original & Hash & Original & Hash\\\\\n\\midrule\n"
        + "\n".join(output) + "\n\\bottomrule\n\\end{tabular}\n")


def cost_assets(generated: Path, figures: Path) -> dict[str, float]:
    path = COST / "summary.json"
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    summary = json.loads(path.read_text())
    assert summary["status"] == "complete"
    for name, expected in summary["input_sha256"].items():
        source = ROOT / name
        assert digest(source) == expected, name
        INPUTS[str(source.relative_to(ROOT))] = expected
    times = rows(COST / "curves.csv")
    ti = {(r["readout"], r["method"], int(r["shortlist_size"])): r for r in times}
    labels = {"target_a": METHOD_NAME, "second_moment_mmd": "MMD", "random": "Random", "exhaustive": "Full validation"}
    selections = [("target_a", 50), ("second_moment_mmd", 50), ("random", 227), ("exhaustive", 455)]
    output = []
    for readout, label in (("ridge", "Ridge"), ("logistic", "Logistic")):
        for method, size in selections:
            t = ti[readout, method, size]
            output.append(f"{label} & {labels[method]} & {size} & {float(t['test_brier_score']):.6f} & "
                          f"{float(t['resident_seconds']):.2f} & {float(t['raw_additive_seconds']):.2f} " + r"\\")
    (generated / "cost_rows.tex").write_text(
        "\\begin{tabular}{llrrrr}\n\\toprule\n"
        + "Readout & Screen & Fits & Brier & Resident (s) & Images (s)\\\\\n\\midrule\n"
        + "\n".join(output) + "\n\\bottomrule\n\\end{tabular}\n")
    output = []
    for readout in ("ridge", "logistic"):
        values = [float(ti[readout, method, size]["cached_two_encoder_batch_seconds"])
                  for method, size in selections]
        output.append(readout.title() + " & " + " & ".join(f"{v:.2f}" for v in values) + r"\\")
    (generated / "cache_batch_rows.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        + "Readout & " + METHOD_NAME + " (50) & MMD (50) & Random (227) & Full (455)\\\\\n\\midrule\n"
        + "\n".join(output) + "\n\\bottomrule\n\\end{tabular}\n")
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.5), layout="constrained")
    colors = {"target_a": "#167D9A", "second_moment_mmd": "#B63855", "random": "#67735B"}
    for i, readout in enumerate(("ridge", "logistic")):
        for j, (field, title) in enumerate((("resident_seconds", "Resident features"),
                                          ("raw_additive_seconds", "Image-cost accounting"))):
            ax = axes[i, j]
            for method in colors:
                budgets = [10, 25, 50, 100, 227]
                ax.plot([float(ti[readout, method, s][field]) for s in budgets],
                        [float(ti[readout, method, s]["test_brier_score"]) for s in budgets],
                        "o-", markersize=3, linewidth=1.1, color=colors[method], label=labels[method])
            ax.plot(float(ti[readout, "exhaustive", 455][field]),
                    float(ti[readout, "exhaustive", 455]["test_brier_score"]),
                    "kx", markersize=5, label="Full (455)")
            ax.set(title=f"{readout.title()}: {title}", xlabel="Mean seconds", ylabel="Final test Brier")
            ax.grid(alpha=.2)
            ax.tick_params(labelsize=8)
            ax.ticklabel_format(axis="y", useOffset=False)
    handles, labels_text = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    fig.savefig(figures / "quality_cost.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    values = {}
    for readout, prefix in (("ridge", "Ridge"), ("logistic", "Logistic")):
        for method, name in (("target_a", "A"), ("second_moment_mmd", "MMD")):
            for field, scope in (("resident_seconds", "Resident"), ("raw_additive_seconds", "Images")):
                values[f"Cost{prefix}{name}{scope}Ratio"] = (
                    float(ti[readout, method, 50][field]) / float(ti[readout, "exhaustive", 455][field]))
    return values


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
        "target_a": METHOD_NAME, "second_moment_mmd": "Second-moment MMD",
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
    supplemental = supplement_numbers()
    readout = readout_assets(generated)
    budget = budget_assets(generated, figures)
    challenge_assets(generated)
    cost = cost_assets(generated, figures)
    readout_precision = {
        "WorstErrorRecall": 0, "WorstErrorGap": 2,
        "ReadoutRidgeA": 2, "ReadoutRidgeMMD": 2,
        "ReadoutLogisticA": 2, "ReadoutLogisticMMD": 2,
        "LogisticSpan": 2, "RidgeSelectedRegret": 6,
    }
    (generated / "numbers.tex").write_text(
        "% Generated from recorded E2b outcome tables.\n"
        + f"\\newcommand{{\\MainRecall}}{{{100 * float(a['mean_true_top_q_recall']):.3f}}}\n"
        + f"\\newcommand{{\\MainRegret}}{{{100 * float(a['mean_selected_normalized_regret']):.6f}}}\n"
        + f"\\newcommand{{\\MainReduction}}{{{100 * float(a['mean_shortlist_reduction']):.3f}}}\n"
        + "% Supplemental diagnostics: post-hoc 2026-09-16 run.\n"
        + "".join(f"\\newcommand{{\\{name}}}{{{value:.2f}}}\n" for name, value in supplemental.items())
        + "% Exploratory projection/readout follow-up, 2026-09-17.\n"
        + "".join(f"\\newcommand{{\\{name}}}{{{value:.{readout_precision.get(name, 3)}f}}}\n"
                  for name, value in readout.items())
        + "% Exploratory decision-budget reanalysis, 2026-09-22.\n"
        + "".join(f"\\newcommand{{\\{name}}}{{{value:.6f}}}\n" for name, value in budget.items())
        + "% Independent matched decision timing and additive image-cost profiles.\n"
        + "".join(f"\\newcommand{{\\{name}}}{{{value:.4f}}}\n" for name, value in cost.items()))
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
    metadata_path = HERE / "submission_metadata.json"
    INPUTS[str(metadata_path.relative_to(ROOT))] = digest(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    (generated / "submission_metadata.tex").write_text(
        "\\newcommand{\\PaperTitleEN}{" + tex_escape(metadata["title_en"]) + "}\n"
        + "\\newcommand{\\PaperTitleZH}{" + tex_escape(metadata["title_zh"]) + "}\n"
        + "\\newcommand{\\PaperKeywordsEN}{" + tex_escape(", ".join(metadata["keywords_en"])) + "}\n",
        encoding="utf-8")
    (HERE / "submission_metadata.md").write_text(
        "# ICLR 2027 投稿信息\n\n"
        "由 `build_assets.py` 从唯一题目、关键词和摘要文本源生成，与论文输入保持一致。\n"
        "状态：作者审阅稿，尚未提交 OpenReview；不表示已完成全部补实验或作者科学核验。\n\n"
        "## Title\n\n" + metadata["title_en"] + "\n\n"
        "## Abstract\n\n" + abstracts["en"] + "\n\n"
        "## Keywords\n\n" + ", ".join(metadata["keywords_en"]) + "\n\n"
        "## 中文题目\n\n" + metadata["title_zh"] + "\n\n"
        "## 中文摘要\n\n" + abstracts["zh"] + "\n\n"
        "## 中文关键词\n\n" + "、".join(metadata["keywords_zh"]) + "\n\n"
        f"英文摘要词数（按空白分词）：{len(abstracts['en'].split())}。\n", encoding="utf-8")
    report = {"input_sha256": INPUTS, "e2b_summary_id": summary["summary_id"],
              "diagnostic_status": "original gates unchanged; exploratory budget and candidate-construction follow-ups reported separately",
              "supplemental_diagnostics_percent": supplemental,
              "readout_followup_percent": readout,
              "decision_budget": budget,
              "decision_cost_ratios": cost,
              "domainnet_brier_spans": spans,
              "primary_metrics": summary["primary_result"]}
    (generated / "asset_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Verified 40 summary rows, 12 complete 455-combination tasks, and equal primary {METHOD_NAME}/MMD metrics.")


if __name__ == "__main__":
    main()
