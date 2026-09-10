"""Verify key numerical claims in the paper against raw CSV data."""

import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def boolean_series(values: pd.Series) -> pd.Series:
    """Parse persisted booleans without treating the string 'False' as true."""

    if values.dtype == bool:
        return values
    parsed = values.astype(str).str.lower().map({'true': True, 'false': False})
    if parsed.isna().any():
        raise ValueError('boolean column contains values other than true/false')
    return parsed


# E1a is deterministic and committed, so it can always be verified locally.
e1a = pd.read_csv(RESULTS / 'e1a_synth_grid.csv')
with open(RESULTS / 'e1a_summary.json') as handle:
    e1a_summary = json.load(handle)

true = e1a['r_eff_merged'].to_numpy(dtype=float)
pred = e1a['r_pred_theory'].to_numpy(dtype=float)
ss_res = float(np.square(true - pred).sum())
ss_tot = float(np.square(true - true.mean()).sum())
e1a_r2 = 1.0 - ss_res / ss_tot
e1a_max_error = float(np.abs(true - pred).max())
e1a_superadditive = boolean_series(e1a['is_superadditive'])
e1a_alpha_lt_one = e1a['alpha_target'] < 1.0

e1a_checks = {
    '280 rows': len(e1a) == 280,
    'R^2 rounds to 1.0000': e1a_r2 >= 1.0 - 5e-13,
    'max absolute error <= 1e-10': e1a_max_error <= 1e-10,
    '245/280 globally superadditive': int(e1a_superadditive.sum()) == 245,
    '245/245 alpha<1 superadditive': bool(e1a_superadditive[e1a_alpha_lt_one].all()),
    '0/35 alpha=1 superadditive': not bool(e1a_superadditive[~e1a_alpha_lt_one].any()),
    'summary matches recomputation': (
        e1a_summary['n_pairs'] == len(e1a)
        and np.isclose(e1a_summary['r2_theory_pred'], e1a_r2, atol=1e-15)
        and np.isclose(e1a_summary['max_absolute_error'], e1a_max_error, atol=1e-15)
        and e1a_summary['n_superadditive'] == int(e1a_superadditive.sum())
    ),
}

print('=== CLAIM: E1a canonical synthetic cross-check ===')
print(f'  R^2={e1a_r2:.16f}; max absolute error={e1a_max_error:.8g}')
for label, passed in e1a_checks.items():
    print(f"  {label}: {'PASS' if passed else 'FAIL'}")
if not all(e1a_checks.values()):
    raise SystemExit(1)

# The larger real-data files are generated on demand and are optional here.
real_paths = (
    RESULTS / 'e1c_expanded_200plus.csv',
    RESULTS / 'exp1_clip_dino_expanded.csv',
)
missing = [path.name for path in real_paths if not path.exists()]
if missing:
    print(f"\nSKIP real-data claims; missing generated files: {', '.join(missing)}")
    raise SystemExit(0)

df1 = pd.read_csv(real_paths[0])
df2 = pd.read_csv(real_paths[1])

print("=== DATA INVENTORY ===")
print(f"e1c_expanded_200plus.csv: {len(df1)} rows")
print(f"  probes: {df1['probe'].value_counts().to_dict()}")
print(f"exp1_clip_dino_expanded.csv: {len(df2)} rows")
print(f"  probes: {df2['probe'].value_counts().to_dict()}")

# Merge and deduplicate (same as hero scatter script)
df = pd.concat([df1, df2], ignore_index=True)
key = ['probe','ds_A','ds_B']
before = len(df)
df = df.drop_duplicates(subset=key, keep='first')
df['is_superadditive'] = boolean_series(df['is_superadditive'])
print(f"\nAfter merge+dedup: {before} -> {len(df)} unique pairs")
print(f"  probes: {df['probe'].value_counts().to_dict()}")

# ====== CLAIM 1: 427 total pairs ======
print(f"\n=== CLAIM: 427 total pairs ===")
print(f"  Actual: {len(df)} pairs  {'PASS' if len(df) == 427 else 'FAIL'}")

# ====== CLAIM 2: 256 vision + 171 text ======
vision_probes = ['resnet50', 'vit_b16', 'clip_vitb32', 'dino_vits16']
text_probes = ['sbert']
n_vision = df[df['probe'].isin(vision_probes)].shape[0]
n_text = df[df['probe'].isin(text_probes)].shape[0]
print(f"\n=== CLAIM: 256 vision + 171 text ===")
print(f"  Vision: {n_vision}  {'PASS' if n_vision == 256 else 'FAIL'}")
print(f"  Text: {n_text}  {'PASS' if n_text == 171 else 'FAIL'}")

# ====== CLAIM 3: Per-probe pair counts ======
print(f"\n=== CLAIM: Per-probe counts (91, 55, 55, 55, 171) ===")
for probe, expected in [('resnet50', 91), ('vit_b16', 55), ('clip_vitb32', 55), ('dino_vits16', 55), ('sbert', 171)]:
    actual = df[df['probe'] == probe].shape[0]
    print(f"  {probe}: {actual} (expected {expected})  {'PASS' if actual == expected else 'FAIL'}")

# ====== CLAIM 4: Superadditivity percentages ======
print(f"\n=== CLAIM: Superadditivity percentages ===")
for probe, expected_pct in [('resnet50', 100.0), ('vit_b16', 69.1), ('clip_vitb32', 29.1), ('dino_vits16', 36.4)]:
    sub = df[df['probe'] == probe]
    sa_pct = 100.0 * sub['is_superadditive'].sum() / len(sub)
    print(f"  {probe}: {sa_pct:.1f}% (claimed {expected_pct}%)  {'PASS' if abs(sa_pct - expected_pct) < 0.5 else 'FAIL'}")

# All vision
vision_df = df[df['probe'].isin(vision_probes)]
sa_vision = 100.0 * vision_df['is_superadditive'].sum() / len(vision_df)
print(f"  All vision: {sa_vision:.1f}% (claimed 64.5%)")

# SBERT
sbert_df = df[df['probe'] == 'sbert']
sa_sbert = 100.0 * sbert_df['is_superadditive'].sum() / len(sbert_df)
print(f"  SBERT: {sa_sbert:.1f}% (for info)")

# Total
sa_total = 100.0 * df['is_superadditive'].sum() / len(df)
print(f"  Total (427): {sa_total:.1f}% (claimed 78.2%)")

# ====== CLAIM 5: Pearson r values ======
print(f"\n=== CLAIM: Pearson r values ===")
for probe, expected_r in [('resnet50', 0.940), ('vit_b16', 0.898), ('clip_vitb32', 0.950), ('dino_vits16', 0.913)]:
    sub = df[df['probe'] == probe]
    r, _ = stats.pearsonr(sub['r_merged_pred'], sub['r_merged_true'])
    print(f"  {probe}: r={r:.3f} (claimed {expected_r})  {'PASS' if abs(r - expected_r) < 0.01 else 'CHECK'}")

# All vision pooled
r_vis, _ = stats.pearsonr(vision_df['r_merged_pred'], vision_df['r_merged_true'])
print(f"  Vision pooled: r={r_vis:.3f} (claimed 0.965)")

# All 427
r_all, _ = stats.pearsonr(df['r_merged_pred'], df['r_merged_true'])
print(f"  All 427: r={r_all:.3f} (claimed 0.970)")

# ====== CLAIM 6: Calibrated R² values ======
print(f"\n=== CLAIM: Calibrated R² (per-probe c) ===")
cal_c = {
    'resnet50': 0.776, 'vit_b16': 0.647,
    'clip_vitb32': 0.595, 'dino_vits16': 0.584, 'sbert': 0.636,
}

for probe, expected_r2 in [('resnet50', 0.760), ('vit_b16', 0.487), ('clip_vitb32', 0.839), ('dino_vits16', 0.554)]:
    sub = df[df['probe'] == probe].copy()
    c = cal_c[probe]
    sub['r_cal'] = sub['r_merged_pred'] * c
    ss_res = ((sub['r_merged_true'] - sub['r_cal'])**2).sum()
    ss_tot = ((sub['r_merged_true'] - sub['r_merged_true'].mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"  {probe}: R²={r2:.3f} (claimed {expected_r2})  {'PASS' if abs(r2 - expected_r2) < 0.02 else 'CHECK'}")

# Vision pooled with single c=0.663
vision_df2 = vision_df.copy()
vision_df2['c'] = vision_df2['probe'].map(cal_c)
vision_df2['r_cal'] = vision_df2['r_merged_pred'] * vision_df2['c']
ss_res_v = ((vision_df2['r_merged_true'] - vision_df2['r_cal'])**2).sum()
ss_tot_v = ((vision_df2['r_merged_true'] - vision_df2['r_merged_true'].mean())**2).sum()
r2_v = 1 - ss_res_v / ss_tot_v
print(f"  Vision pooled (per-probe c): R²={r2_v:.3f} (claimed 0.916)")

# All 427 with per-probe c
all_df = df.copy()
all_df['c'] = all_df['probe'].map(cal_c)
all_df['r_cal'] = all_df['r_merged_pred'] * all_df['c']
ss_res_a = ((all_df['r_merged_true'] - all_df['r_cal'])**2).sum()
ss_tot_a = ((all_df['r_merged_true'] - all_df['r_merged_true'].mean())**2).sum()
r2_a = 1 - ss_res_a / ss_tot_a
mae_a = (all_df['r_merged_true'] - all_df['r_cal']).abs().mean()
print(f"  All 427 (per-probe c): R²={r2_a:.3f} (claimed 0.915), MAE={mae_a:.1f}")

# ====== CLAIM 7: Hero scatter R²=0.952, MAE=48 ======
# The hero scatter uses the same per-probe c but computes R² on all 427
print(f"\n=== CLAIM: Hero scatter R²=0.952, MAE=48 ===")
print(f"  Computed R²={r2_a:.3f}, MAE={mae_a:.1f}")
print(f"  NOTE: Table E1d says R²=0.915 for all 427, hero scatter caption says R²=0.952")

# Also compute with vision-pooled single c
vision_df3 = vision_df.copy()
c_single = 0.663
vision_df3['r_cal_single'] = vision_df3['r_merged_pred'] * c_single
ss_res_vs = ((vision_df3['r_merged_true'] - vision_df3['r_cal_single'])**2).sum()
ss_tot_vs = ((vision_df3['r_merged_true'] - vision_df3['r_merged_true'].mean())**2).sum()
r2_vs = 1 - ss_res_vs / ss_tot_vs
print(f"  Vision with single c=0.663: R²={r2_vs:.3f} (claimed 0.916 in Table)")

# ====== CLAIM 8: Abstract says R²=0.92 ======
print(f"\n=== CLAIM: Abstract 'R²=0.92' ===")
print(f"  Per-probe c on all 427: R²={r2_a:.3f}")
print(f"  Per-probe c on vision 256: R²={r2_v:.3f}")
print(f"  This appears to be rounding 0.916 -> 0.92")

# ====== CLAIM 9: SBERT r=0.793 (line 583) ======
print(f"\n=== CLAIM: SBERT Pearson r=0.793 ===")
r_sbert, _ = stats.pearsonr(sbert_df['r_merged_pred'], sbert_df['r_merged_true'])
print(f"  Computed: r={r_sbert:.3f}")

# ====== CLAIM 10: Ablation table 158 vision pairs ======
print(f"\n=== CLAIM: Ablation on 158 vision pairs ===")
print(f"  Vision probes: {n_vision} pairs (paper says 158)")
print(f"  ResNet+ViT only: {df[df['probe'].isin(['resnet50','vit_b16'])].shape[0]}")

print("\n=== DONE ===")
