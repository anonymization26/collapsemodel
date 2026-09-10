"""
experiment_e5_joint_matrix.py -- E5: Joint Evaluation Matrix Validation
========================================================================
Phase 5 (PLAN.md §5): Show the connections across all four experiments
in a unified (r_eff, SA_k) coordinate system.

Core contributions:
  1. Joint coordinate plot: (r_eff^{1/2}, SA_k) -- x=richness, y=alignment, color=result
  2. Joint score Score(M,D) = r_eff(D)^0.5 * SA_k(src->D)
  3. Spearman rho comparison: SA_k alone vs r_eff alone vs Score (joint)
  4. Four-experiment panel: E1 (merge gain) + E2 (transferability) + E3 (selection order) + E4 (domain distance)

Data sources:
  - E2 cache (e2_scores.csv): LP accuracy + SA_k + LogME/H-score/TransRate
  - E1b cache (e1b_real_pairs.csv): r_eff + delta_r + SA_k
  - E4 cache (e4_domain_distance_resnet50.csv): domain distance matrix
  - E3 cache (e3_selection_order.csv, e3_reff_trajectory.csv): greedy selection order

Run: python experiment_e5_joint_matrix.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

# ── Paths ─────────────────────────────────────────────────────────────────
# repro/scripts/ -> BASE_DIR points to the repro/ root
import os as _os
BASE_DIR  = Path(_os.environ.get(
    "COLLAPSE_REPRO_ROOT",
    Path(__file__).resolve().parent.parent
)).resolve()
RES_DIR   = BASE_DIR / "results"
FIG_DIR   = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# 1. Load experiment data
# ─────────────────────────────────────────────────────────────────────────

def load_data():
    """Load and consolidate E1b / E2 / E3 / E4 data."""
    # E2: transferability prediction
    e2 = pd.read_csv(RES_DIR / "e2_scores.csv")

    # E1b: r_eff merge results for real dataset pairs
    e1b = pd.read_csv(RES_DIR / "e1b_real_pairs.csv")

    # E3: greedy selection order
    e3_order = pd.read_csv(RES_DIR / "e3_selection_order.csv")
    e3_traj  = pd.read_csv(RES_DIR / "e3_reff_trajectory.csv")

    # E4: domain distance matrix
    e4_r50  = pd.read_csv(RES_DIR / "e4_domain_distance_resnet50.csv", index_col=0)
    e4_vit  = pd.read_csv(RES_DIR / "e4_domain_distance_vit_b16.csv",  index_col=0)

    # E4 gain consistency
    e4_gain = pd.read_csv(RES_DIR / "e4_gain_consistency.csv")

    return e2, e1b, e3_order, e3_traj, e4_r50, e4_vit, e4_gain


# ─────────────────────────────────────────────────────────────────────────
# 2. Build joint evaluation matrix
# ─────────────────────────────────────────────────────────────────────────

def build_joint_matrix(e2: pd.DataFrame, e1b: pd.DataFrame):
    """
    Each row of the joint matrix = one (probe, target_dataset) evaluation point.
    Columns: r_eff_target, sa_k, lp_acc, joint_score, logme, hscore, transrate
    """
    rows = []
    for _, r in e2.iterrows():
        probe  = r["probe"]
        tgt    = r["target_ds"]
        sa_k   = r["sa_k"]
        lp_acc = r["lp_acc"]
        logme  = r.get("logme", np.nan)
        hscore = r.get("hscore", np.nan)
        transrate = r.get("transrate", np.nan)

        # Find r_eff_target from E1b (= r_eff_B when ds_A=cifar10)
        mask = (e1b["probe"] == probe) & (e1b["ds_B"] == tgt)
        if mask.sum() == 0:
            # Try ds_A
            mask = (e1b["probe"] == probe) & (e1b["ds_A"] == tgt)
            col_reff = "r_eff_A"
        else:
            col_reff = "r_eff_B"

        if mask.sum() > 0:
            r_eff_tgt = e1b.loc[mask, col_reff].values[0]
        else:
            r_eff_tgt = np.nan

        # Joint score: richness x alignment
        joint_score = np.sqrt(r_eff_tgt) * sa_k if not np.isnan(r_eff_tgt) else np.nan

        rows.append({
            "probe":        probe,
            "target_ds":    tgt,
            "r_eff_target": r_eff_tgt,
            "sa_k":         sa_k,
            "lp_acc":       lp_acc,
            "joint_score":  joint_score,
            "logme":        logme,
            "hscore":       hscore,
            "transrate":    transrate,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# 3. Compute per-metric correlation (Spearman rho vs LP accuracy)
# ─────────────────────────────────────────────────────────────────────────

def compute_predictor_comparison(jm: pd.DataFrame):
    """Compare Spearman rho for SA_k / r_eff^0.5 / Joint / LogME / H-score / TransRate."""
    y = jm["lp_acc"].values
    metrics = {
        r"SA$_k$ (ours, no labels)":           jm["sa_k"].values,
        r"$r_{\rm eff}^{1/2}$ alone":          np.sqrt(jm["r_eff_target"].values),
        r"Joint = $r_{\rm eff}^{1/2}\times$ SA$_k$": jm["joint_score"].values,
        r"LogME (needs labels)":               jm["logme"].values,
        r"H-score (needs labels)":             jm["hscore"].values,
        r"TransRate (needs labels)":           jm["transrate"].values,
    }

    rows = []
    for name, x in metrics.items():
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 2:
            rows.append({"metric": name, "spearman_rho": np.nan, "pearson_r": np.nan, "n": valid.sum()})
            continue
        rho, p_s = spearmanr(x[valid], y[valid])
        r,   p_p = pearsonr(x[valid], y[valid])
        rows.append({"metric": name, "spearman_rho": rho, "pearson_r": r, "n": valid.sum()})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# 4. Visualization: comprehensive four-experiment panel
# ─────────────────────────────────────────────────────────────────────────

PROBE_COLOR = {"resnet50": "#2196F3", "vit_b16": "#FF5722"}
PROBE_MARKER = {"resnet50": "o", "vit_b16": "^"}
DS_LABEL = {
    "cifar10": "CIFAR-10", "svhn": "SVHN",
    "stl10":   "STL-10",   "mnist": "MNIST",
}

def plot_comprehensive(jm, pred_df, e1b, e3_order, e3_traj, e4_r50, e4_vit, e4_gain):
    """
    6-subplot comprehensive panel:
      [0,0] Joint coordinate plot (r_eff^0.5 x SA_k, colored by LP accuracy)
      [0,1] Predictor comparison bar chart (Spearman rho)
      [1,0] E1b: merge gain delta_r vs SA_k (with theory prediction)
      [1,1] E3: cumulative r_eff trajectory for two greedy strategies
      [2,0] E4: SA_k heatmap (ResNet50)
      [2,1] E4: cross-probe domain distance consistency scatter
    """
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.45, wspace=0.38,
                           left=0.08, right=0.97, top=0.94, bottom=0.07)

    # ── (0,0) Joint coordinate plot ───────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    sc = ax00.scatter(
        np.sqrt(jm["r_eff_target"]),
        jm["sa_k"],
        c=jm["lp_acc"],
        cmap="RdYlGn",
        s=160, zorder=3,
        edgecolors="k", linewidths=0.8,
        vmin=0.3, vmax=1.0
    )
    for _, row in jm.iterrows():
        lbl = f"{DS_LABEL.get(row['target_ds'], row['target_ds'])}\n({row['probe'][:3].upper()})"
        ax00.annotate(lbl,
                      (np.sqrt(row["r_eff_target"]), row["sa_k"]),
                      textcoords="offset points", xytext=(6, 4),
                      fontsize=6.5, color="#333333")
    cb = plt.colorbar(sc, ax=ax00, pad=0.02)
    cb.set_label("LP Accuracy", fontsize=8)
    ax00.set_xlabel(r"$r_{\rm eff}^{1/2}$ (representation richness)", fontsize=9)
    ax00.set_ylabel(r"SA$_k$ (alignment to CIFAR-10)", fontsize=9)
    ax00.set_title("E5a: Joint Coordinate Map\n"
                   r"(high $r_{\rm eff}$ + high SA$_k$ = best transfer)", fontsize=9)
    ax00.grid(True, alpha=0.3)

    # ── (0,1) Predictor comparison ────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    valid_pred = pred_df.dropna(subset=["spearman_rho"])
    colors_bar = ["#1976D2" if "ours" in m or "Joint" in m else
                  "#90A4AE" for m in valid_pred["metric"]]
    bars = ax01.barh(valid_pred["metric"], valid_pred["spearman_rho"],
                     color=colors_bar, edgecolor="white", height=0.55)
    ax01.axvline(0, color="k", linewidth=0.8)
    ax01.set_xlim(-1.2, 1.2)
    ax01.set_xlabel("Spearman ρ vs LP Accuracy", fontsize=9)
    ax01.set_title("E5b: Predictor Comparison\n(no-label methods in blue)", fontsize=9)
    for bar, rho in zip(bars, valid_pred["spearman_rho"]):
        ax01.text(rho + 0.03 if rho >= 0 else rho - 0.03,
                  bar.get_y() + bar.get_height() / 2,
                  f"{rho:+.2f}", va="center", ha="left" if rho >= 0 else "right",
                  fontsize=8, fontweight="bold")
    ax01.grid(True, axis="x", alpha=0.3)
    ax01.tick_params(axis="y", labelsize=7.5)

    # ── (1,0) E1b delta_r vs SA_k ────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    for probe, grp in e1b.groupby("probe"):
        ax10.scatter(grp["sa_k"], grp["delta_r"],
                     c=PROBE_COLOR[probe], marker=PROBE_MARKER[probe],
                     s=100, label=probe, zorder=3, edgecolors="k", linewidths=0.6)
        # Annotate dataset pairs
        for _, row in grp.iterrows():
            lbl = f"{DS_LABEL.get(row['ds_A'],row['ds_A'])}\n+{DS_LABEL.get(row['ds_B'],row['ds_B'])}"
            ax10.annotate(lbl, (row["sa_k"], row["delta_r"]),
                          textcoords="offset points", xytext=(5, 3), fontsize=6)
    ax10.axhline(0, color="k", linestyle="--", linewidth=0.9, alpha=0.6,
                 label="Δr = 0 (additive)")
    ax10.set_xlabel(r"SA$_k$ (subspace alignment)", fontsize=9)
    ax10.set_ylabel(r"$\Delta r = r_{\rm merged} - r_{\rm dom}$", fontsize=9)
    ax10.set_title("E1b: Merge Gain vs. Alignment\n(above dashed = superadditive)", fontsize=9)
    ax10.legend(fontsize=7.5)
    ax10.grid(True, alpha=0.3)

    # ── (1,1) E3 greedy trajectory ────────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    # e3_traj columns: step, reff_delta_dir, reff_reff_base, reff_random
    steps = e3_traj["step"].values
    ax11.plot(steps, e3_traj["reff_delta_dir"].values,
              marker="o", ls="-",  color="#1976D2", linewidth=2.0,
              label=r"$\Delta_{\rm dir}$-greedy (ours)", zorder=3)
    ax11.plot(steps, e3_traj["reff_reff_base"].values,
              marker="s", ls="--", color="#FF7043", linewidth=2.0,
              label=r"$r_{\rm eff}$-greedy", zorder=3)
    if "reff_random" in e3_traj.columns:
        ax11.plot(steps, e3_traj["reff_random"].values,
                  marker="x", ls=":", color="#9E9E9E", linewidth=1.5,
                  label="Random", zorder=2)
    # Annotate datasets selected by the Delta_dir strategy
    for _, row in e3_order.iterrows():
        rnd = row["round"]
        ds_a = DS_LABEL.get(str(row["strategy_a_ds"]).lower(), row["strategy_a_ds"])
        ds_b = DS_LABEL.get(str(row["strategy_b_ds"]).lower(), row["strategy_b_ds"])
        y_a = e3_traj.loc[e3_traj["step"] == rnd, "reff_delta_dir"].values
        y_b = e3_traj.loc[e3_traj["step"] == rnd, "reff_reff_base"].values
        if len(y_a):
            ax11.annotate(ds_a, (rnd, y_a[0]),
                          textcoords="offset points", xytext=(5, 5), fontsize=7, color="#1976D2")
        if len(y_b):
            ax11.annotate(ds_b, (rnd, y_b[0]),
                          textcoords="offset points", xytext=(5, -13), fontsize=7, color="#FF7043")

    ax11.set_xlabel("Selection Round", fontsize=9)
    ax11.set_ylabel(r"Cumulative $r_{\rm eff}$ added", fontsize=9)
    ax11.set_title("E3: Greedy Selection Trajectory\n(faster diversity = better early rounds)", fontsize=9)
    ax11.legend(fontsize=7.5)
    ax11.grid(True, alpha=0.3)

    # ── (2,0) E4 SA_k heatmap ─────────────────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    ds_order = ["cifar10", "mnist", "stl10", "svhn"]
    ds_labels_hm = [DS_LABEL.get(d, d) for d in ds_order]
    # Keep only datasets present in the matrix
    avail = [d for d in ds_order if d in e4_r50.index]
    mat   = e4_r50.loc[avail, avail].values.astype(float)
    avail_lbl = [DS_LABEL.get(d, d) for d in avail]

    im = ax20.imshow(mat, cmap="YlOrRd_r", vmin=0.0, vmax=1.0)
    ax20.set_xticks(range(len(avail))); ax20.set_xticklabels(avail_lbl, fontsize=8)
    ax20.set_yticks(range(len(avail))); ax20.set_yticklabels(avail_lbl, fontsize=8)
    for i in range(len(avail)):
        for j in range(len(avail)):
            ax20.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                      fontsize=8, fontweight="bold" if i != j else "normal",
                      color="white" if mat[i,j] < 0.3 else "black")
    plt.colorbar(im, ax=ax20, label=r"SA$_k$", pad=0.02)
    ax20.set_title(r"E4: Domain Distance Matrix (ResNet50)" + "\n" +
                   r"lower SA$_k$ = more complementary", fontsize=9)

    # ── (2,1) E4 cross-probe consistency ─────────────────────────────
    ax21 = fig.add_subplot(gs[2, 1])
    try:
        # gain_consistency has sa_k (resnet50) and sa_k_vit (vit_b16)
        gd = e4_gain.dropna()
        # find common pairs between the two probes
        common_ds = list(set(e4_r50.columns) & set(e4_vit.columns))
        pairs_r50 = []
        pairs_vit = []
        pair_labels = []
        for i, da in enumerate(common_ds):
            for j, db in enumerate(common_ds):
                if i < j:
                    pairs_r50.append(e4_r50.loc[da, db])
                    pairs_vit.append(e4_vit.loc[da, db])
                    pair_labels.append(f"{DS_LABEL.get(da,da)}\n{DS_LABEL.get(db,db)}")

        pairs_r50 = np.array(pairs_r50)
        pairs_vit = np.array(pairs_vit)

        if len(pairs_r50) >= 2:
            r_val, _ = pearsonr(pairs_r50, pairs_vit)
        else:
            r_val = np.nan

        ax21.scatter(pairs_r50, pairs_vit,
                     c="#7B1FA2", s=120, zorder=3, edgecolors="k", linewidths=0.7)
        for k_, lbl in enumerate(pair_labels):
            ax21.annotate(lbl, (pairs_r50[k_], pairs_vit[k_]),
                          textcoords="offset points", xytext=(5, 4), fontsize=7)

        # Fit line
        if len(pairs_r50) >= 2 and not np.isnan(r_val):
            z = np.polyfit(pairs_r50, pairs_vit, 1)
            xfit = np.linspace(pairs_r50.min()-0.02, pairs_r50.max()+0.02, 100)
            ax21.plot(xfit, np.polyval(z, xfit), "k--", linewidth=1.2, alpha=0.6)
            ax21.text(0.05, 0.92, f"Pearson r = {r_val:+.2f}",
                      transform=ax21.transAxes, fontsize=9, fontweight="bold",
                      bbox=dict(boxstyle="round", fc="lightyellow", ec="gray", alpha=0.8))
    except Exception as e:
        ax21.text(0.5, 0.5, f"Cross-probe plot\n(see E4 for full data)\n{e}",
                  ha="center", va="center", transform=ax21.transAxes, fontsize=8)

    ax21.set_xlabel(r"SA$_k$ (ResNet50)", fontsize=9)
    ax21.set_ylabel(r"SA$_k$ (ViT-B/16)", fontsize=9)
    ax21.set_title("E4: Cross-Probe Consistency\n(domain geometry is encoder-independent)", fontsize=9)
    ax21.grid(True, alpha=0.3)

    # ── Global title ──────────────────────────────────────────────────
    fig.suptitle(
        "E5: Joint Evaluation Matrix — Unified View of the Grassmann Framework\n"
        r"Score$(M,\mathcal{D}) = r_{\rm eff}^{1/2} \times$ SA$_k$ unifies E1–E4",
        fontsize=11, fontweight="bold", y=0.98
    )

    out = FIG_DIR / "e5_joint_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out}")
    return out


def _plot_fallback_e3(ax, dir_data, reff_data):
    """Fallback E3 trajectory plot (manual cumulative r_eff)."""
    # r_eff reference values
    reff_map = {"cifar10": 344.3, "stl10": 357.6, "svhn": 244.5, "mnist": 202.0}

    for data, ls, color, lbl in [
        (dir_data,  "-",  "#1976D2", r"$\Delta_{\rm dir}$-greedy (ours)"),
        (reff_data, "--", "#FF7043", r"$r_{\rm eff}$-greedy"),
    ]:
        if len(data) == 0:
            continue
        # Sort by round and accumulate
        data_s = data.sort_values("round") if "round" in data.columns else data
        cum = 0.0
        rounds, cums = [0], [0]
        for _, row in data_s.iterrows():
            ds = str(row.get("dataset","")).lower().replace("-","")
            cum += reff_map.get(ds, 200)
            rounds.append(row.get("round", len(rounds)))
            cums.append(cum)
        ax.plot(rounds, cums, marker="o", ls=ls, color=color, linewidth=2.0, label=lbl)


# ─────────────────────────────────────────────────────────────────────────
# 5. Joint score analysis (Table)
# ─────────────────────────────────────────────────────────────────────────

def print_joint_analysis(jm: pd.DataFrame, pred_df: pd.DataFrame):
    print("\n" + "="*70)
    print("E5: Joint Evaluation Matrix -- (r_eff^0.5, SA_k, LP_acc)")
    print("="*70)

    disp = jm[["probe","target_ds","r_eff_target","sa_k","joint_score","lp_acc"]].copy()
    disp["r_eff_sqrt"] = np.sqrt(disp["r_eff_target"])
    print(disp[["probe","target_ds","r_eff_sqrt","sa_k","joint_score","lp_acc"]].to_string(index=False))

    print("\n" + "-"*70)
    print("Spearman rho per predictor vs LP accuracy:")
    print("-"*70)
    for _, row in pred_df.iterrows():
        rho = row["spearman_rho"]
        marker = "* no labels" if "ours" in row["metric"] or "Joint" in row["metric"] else "  needs labels"
        print(f"  {marker}  {row['metric']:<42s}  rho={rho:+.3f}  (n={row['n']:.0f})")

    # Joint score advantage analysis
    joint_rho_row = pred_df[pred_df["metric"].str.contains("Joint")]
    sa_rho_row    = pred_df[pred_df["metric"].str.contains("SA\\$_k\\$")]
    reff_rho_row  = pred_df[pred_df["metric"].str.contains("r_\\{\\\\rm eff\\}")]

    print("\n" + "-"*70)
    print("Improvement of joint score over individual metrics:")
    if len(joint_rho_row) > 0 and len(sa_rho_row) > 0:
        delta_sa   = joint_rho_row["spearman_rho"].values[0] - sa_rho_row["spearman_rho"].values[0]
        print(f"  Joint vs SA_k alone:    Δρ = {delta_sa:+.3f}")
    if len(joint_rho_row) > 0 and len(reff_rho_row) > 0:
        delta_reff = joint_rho_row["spearman_rho"].values[0] - reff_rho_row["spearman_rho"].values[0]
        print(f"  Joint vs r_eff alone:   Δρ = {delta_reff:+.3f}")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────
# 6. Save results
# ─────────────────────────────────────────────────────────────────────────

def save_results(jm: pd.DataFrame, pred_df: pd.DataFrame):
    jm.to_csv(RES_DIR / "e5_joint_matrix.csv", index=False)
    pred_df.to_csv(RES_DIR / "e5_predictor_comparison.csv", index=False)
    print(f"Results saved: {RES_DIR / 'e5_joint_matrix.csv'}")
    print(f"Results saved: {RES_DIR / 'e5_predictor_comparison.csv'}")


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()
    print("=" * 60)
    print("E5: Joint Evaluation Matrix Validation")
    print("=" * 60)

    # 1. Load data
    e2, e1b, e3_order, e3_traj, e4_r50, e4_vit, e4_gain = load_data()
    print(f"  E2 records: {len(e2)}   E1b records: {len(e1b)}")

    # 2. Build joint matrix
    jm = build_joint_matrix(e2, e1b)
    print(f"  Joint matrix: {len(jm)} evaluation points")

    # 3. Predictor comparison
    pred_df = compute_predictor_comparison(jm)

    # 4. Print analysis
    print_joint_analysis(jm, pred_df)

    # 5. Visualization
    fig_path = plot_comprehensive(jm, pred_df, e1b, e3_order, e3_traj, e4_r50, e4_vit, e4_gain)

    # 6. Save
    save_results(jm, pred_df)

    elapsed = (time.time() - t0) / 60
    print(f"\nE5 done  elapsed: {elapsed:.1f} min")
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
