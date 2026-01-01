"""Compute updated ablation table on full 256 vision pairs (+ 427 total)."""
import pandas as pd
import numpy as np
from scipy import stats

df1 = pd.read_csv('results/e1c_expanded_200plus.csv')
df2 = pd.read_csv('results/exp1_clip_dino_expanded.csv')
df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=['probe','ds_A','ds_B'], keep='first')

vision_probes = ['resnet50', 'vit_b16', 'clip_vitb32', 'dino_vits16']

# Compute predictors
df['r_sum'] = df['r_A'] + df['r_B']
df['r_max'] = df[['r_A','r_B']].max(axis=1)
df['one_minus_alpha'] = 1 - df['alpha']

print("=== ABLATION: Spearman rho between predictor and true merged r_eff ===\n")

for label, probes in [
    ('ResNet-50 (n=91)', ['resnet50']),
    ('ViT-B/16 (n=55)', ['vit_b16']),
    ('CLIP ViT-B/32 (n=55)', ['clip_vitb32']),
    ('DINO ViT-S/16 (n=55)', ['dino_vits16']),
    ('All vision (n=256)', vision_probes),
    ('All 427', vision_probes + ['sbert']),
]:
    sub = df[df['probe'].isin(probes)]
    n = len(sub)
    print(f"{label} [actual n={n}]")
    for pred_name, pred_col in [
        ('Collapse Model', 'r_merged_pred'),
        ('r_A + r_B', 'r_sum'),
        ('max(r_A, r_B)', 'r_max'),
        ('1-alpha (V only)', 'one_minus_alpha'),
        ('gamma (Sigma only)', 'gamma'),
    ]:
        rho, p = stats.spearmanr(sub[pred_col], sub['r_merged_true'])
        print(f"  {pred_name:25s}: rho={rho:.3f}  (p={p:.2e})")
    print()
