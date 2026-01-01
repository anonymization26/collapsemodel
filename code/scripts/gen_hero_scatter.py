#!/usr/bin/env python3
"""Generate hero scatter figure: predicted vs actual merged r_eff (427 pairs)."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# --- Load & merge data ---
import os as _os
base = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results")
df1 = pd.read_csv(_os.path.join(base, 'e1c_expanded_200plus.csv'))
df2 = pd.read_csv(_os.path.join(base, 'exp1_clip_dino_expanded.csv'))

df = pd.concat([df1, df2], ignore_index=True)
df = df.drop_duplicates(subset=['probe', 'ds_A', 'ds_B'], keep='first')
print(f"Total unique pairs: {len(df)}")
print(df['probe'].value_counts())

# --- Calibration constants (from Table 1 / E1d) ---
cal_c = {
    'resnet50': 0.776,
    'vit_b16': 0.647,
    'clip_vitb32': 0.595,
    'dino_vits16': 0.584,
    'sbert': 0.636,
}

df['c'] = df['probe'].map(cal_c)
df['r_merged_cal'] = df['r_merged_pred'] * df['c']

actual = df['r_merged_true'].values
pred_raw = df['r_merged_pred'].values
pred_cal = df['r_merged_cal'].values

# --- Metrics ---
r_raw, _ = stats.pearsonr(actual, pred_raw)
r_cal, _ = stats.pearsonr(actual, pred_cal)
ss_res_cal = np.sum((actual - pred_cal)**2)
ss_tot = np.sum((actual - np.mean(actual))**2)
r2_cal = 1 - ss_res_cal / ss_tot
mae_cal = np.mean(np.abs(actual - pred_cal))

print(f"Uncalibrated Pearson r = {r_raw:.3f}")
print(f"Calibrated   Pearson r = {r_cal:.3f}, R² = {r2_cal:.3f}, MAE = {mae_cal:.1f}")

# --- Probe display config ---
probe_cfg = {
    'resnet50':     {'label': 'ResNet-50',   'color': '#2176AE', 'marker': 'o'},
    'vit_b16':      {'label': 'ViT-B/16',    'color': '#E8451E', 'marker': 's'},
    'clip_vitb32':  {'label': 'CLIP ViT-B/32','color': '#57A773', 'marker': '^'},
    'dino_vits16':  {'label': 'DINO ViT-S/16','color': '#8B5CF6', 'marker': 'D'},
    'sbert':        {'label': 'SBERT',        'color': '#F59E0B', 'marker': 'v'},
}

# --- Figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

for probe_name, cfg in probe_cfg.items():
    mask = df['probe'] == probe_name
    sub = df[mask]
    n = len(sub)
    
    # Panel a: uncalibrated
    ax1.scatter(sub['r_merged_true'], sub['r_merged_pred'],
                c=cfg['color'], marker=cfg['marker'], s=18, alpha=0.7,
                edgecolors='white', linewidths=0.3,
                label=f"{cfg['label']} ({n})")
    
    # Panel b: calibrated
    ax2.scatter(sub['r_merged_true'], sub['r_merged_cal'],
                c=cfg['color'], marker=cfg['marker'], s=18, alpha=0.7,
                edgecolors='white', linewidths=0.3,
                label=f"{cfg['label']} ({n})")

# Identity lines
for ax in [ax1, ax2]:
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.4, zorder=0)
    ax.set_xlim(0, lims[1])
    ax.set_ylim(0, lims[1])
    ax.set_xlabel('Actual merged $r_{\\mathrm{eff}}$', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_aspect('equal')

ax1.set_ylabel('Predicted merged $r_{\\mathrm{eff}}$', fontsize=10)
ax2.set_ylabel('Calibrated predicted $r_{\\mathrm{eff}}$', fontsize=10)

ax1.set_title(f'(a) Uncalibrated  (Pearson $r$={r_raw:.3f})', fontsize=10)
ax2.set_title(f'(b) Calibrated  ($R^2$={r2_cal:.3f}, MAE={mae_cal:.0f})', fontsize=10)

# Legend
ax2.legend(fontsize=7, loc='upper left', framealpha=0.9, handlelength=1.2)

plt.tight_layout(w_pad=2.0)
out = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "figures", "fig_hero_scatter.pdf")
fig.savefig(out, bbox_inches='tight', pad_inches=0.05)
print(f"Saved to {out}")
plt.close()
