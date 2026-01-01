"""
Experiment 1: Extend CLIP/DINO to all 14 vision datasets.
Extract features and compute all C(14,2)=91 pairs.
"""
import os, sys, csv, math, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR, SEED, DATA_DIR

RESULTS = str(RESULT_DIR)
N_SAMPLES = 2000
K = 20

# ---- ImageNet loader (same as run_e1c_expanded_200.py) ----
IMAGENET_GROUPS = {
    "imagenet_animal":   list(range(0, 398)),
    "imagenet_vehicle":  list(range(404, 547)),
    "imagenet_artifact": list(range(547, 700)),
    "imagenet_food":     list(range(924, 970)),
    "imagenet_scene":    list(range(700, 924)),
}

def load_imagenet_val_features(model, forward_fn, group_name, class_ids, n_samples=2000, device="cuda"):
    import torchvision.transforms as T
    from PIL import Image
    val_dir = os.path.join(str(DATA_DIR), "ILSVRC2012_img_val")
    gt_file = os.path.join(str(DATA_DIR), "ILSVRC2012_validation_ground_truth.txt")
    with open(gt_file) as f:
        val_labels = [int(x.strip()) - 1 for x in f.readlines()]
    all_images = sorted([f for f in os.listdir(val_dir) if f.endswith('.JPEG')])
    class_set = set(class_ids)
    indices = [i for i, lbl in enumerate(val_labels) if lbl in class_set]
    rng = np.random.default_rng(SEED)
    if len(indices) > n_samples:
        indices = rng.choice(indices, size=n_samples, replace=False).tolist()
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                           T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    feats, labs = [], []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            batch_idx = indices[start:start+128]
            imgs = []
            for idx in batch_idx:
                img = Image.open(os.path.join(val_dir, all_images[idx])).convert('RGB')
                imgs.append(transform(img))
                labs.append(val_labels[idx])
            batch = torch.stack(imgs).to(device)
            feat = forward_fn(batch)
            feats.append(feat.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32), np.array(labs, dtype=np.int64)

# ---- Metrics ----
def effective_rank(H):
    sv = np.linalg.svd(H, compute_uv=False)
    sv = sv[sv > 1e-10]
    p = sv / sv.sum()
    return math.exp(-np.sum(p * np.log(p + 1e-30)))

def nuclear_norm(H):
    return np.linalg.svd(H, compute_uv=False).sum()

def subspace_alignment(H_A, H_B, k=K):
    """Return (scalar SA_k, per-dim cos²θᵢ array of length k)."""
    _, _, V_A = np.linalg.svd(H_A, full_matrices=False)
    _, _, V_B = np.linalg.svd(H_B, full_matrices=False)
    V_Ak = V_A[:k].T
    V_Bk = V_B[:k].T
    M = V_Ak.T @ V_Bk
    sv = np.linalg.svd(M, compute_uv=False)
    cos2 = sv ** 2                          # shape (k,), sorted descending
    return float(np.mean(cos2)), cos2

def collapse_predict(r_A, r_B, gamma, alpha):
    r_dom = r_A if gamma <= 1 else r_B
    r_sub = r_B if gamma <= 1 else r_A
    A = 1 + gamma**2
    D = math.sqrt((1 - gamma**2)**2 + 4 * gamma**2 * alpha)
    c_plus = math.sqrt((A + D) / 2)
    c_minus = math.sqrt(max((A - D) / 2, 1e-30))
    r_star = c_plus / (c_plus + c_minus)
    eps = 1e-15
    r_star = max(min(r_star, 1 - eps), eps)
    H_b = -r_star * math.log(r_star) - (1 - r_star) * math.log(1 - r_star)
    log_r = H_b + r_star * math.log(max(r_dom, 1e-10)) + (1 - r_star) * math.log(max(r_sub, 1e-10))
    rho_c = math.exp(H_b / max(1 - r_star, eps))
    return math.exp(log_r), r_star, rho_c

def l2_normalize_rows(H):
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    return H / np.maximum(norms, 1e-10)

def compute_pair(H_A, H_B):
    H_A = l2_normalize_rows(H_A)
    H_B = l2_normalize_rows(H_B)
    rng = np.random.default_rng(SEED)
    if H_A.shape[0] > N_SAMPLES:
        H_A = H_A[rng.choice(H_A.shape[0], N_SAMPLES, replace=False)]
    if H_B.shape[0] > N_SAMPLES:
        H_B = H_B[rng.choice(H_B.shape[0], N_SAMPLES, replace=False)]
    r_A = effective_rank(H_A)
    r_B = effective_rank(H_B)
    gamma = nuclear_norm(H_B) / nuclear_norm(H_A)
    alpha, cos2_vec = subspace_alignment(H_A, H_B)   # unpack scalar + vector
    H_merged = np.concatenate([H_A, H_B], axis=0)
    r_merged_true = effective_rank(H_merged)
    r_merged_pred, r_star, rho_c = collapse_predict(r_A, r_B, gamma, alpha)
    r_dom = r_A if gamma <= 1 else r_B
    r_sub = r_B if gamma <= 1 else r_A
    # Per-dimension cos²θᵢ stored as cos2_01 … cos2_20 (1-indexed)
    cos2_cols = {f'cos2_{i+1:02d}': float(cos2_vec[i]) for i in range(len(cos2_vec))}
    return {
        'r_A': r_A, 'r_B': r_B, 'r_merged_true': r_merged_true,
        'r_merged_pred': r_merged_pred, 'alpha': alpha,
        'v_alpha': float(np.var(cos2_vec)),             # alignment variance
        'gamma': gamma,
        'nuclear_A': nuclear_norm(H_A), 'nuclear_B': nuclear_norm(H_B),
        'r_star': r_star, 'rho_c': rho_c, 'ratio': r_dom/max(r_sub,1e-10),
        'delta_r': r_merged_true - r_dom,
        'pred_error_pct': abs(r_merged_pred - r_merged_true) / r_merged_true * 100,
        'is_superadditive': r_merged_true > r_dom,
        **cos2_cols,                                    # cos2_01 … cos2_20
    }

def get_domain(ds):
    if ds.startswith("imagenet_"): return "imagenet"
    dm = {"cifar10":"natural","cifar100":"natural","stl10":"natural","svhn":"digit",
          "mnist":"handwritten","fashion_mnist":"handwritten","bloodmnist":"medical",
          "dermamnist":"medical","pathmnist":"medical"}
    return dm.get(ds, "other")

# ---- Build CLIP and DINO models ----
def build_clip_vitb32(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.visual.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, preprocess

def build_dino_vits16(device):
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
    model = model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model

# ---- Main ----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    VISION_DS = ["cifar10","cifar100","stl10","svhn","mnist","fashion_mnist",
                 "bloodmnist","dermamnist","pathmnist"]
    IMAGENET_DS = list(IMAGENET_GROUPS.keys())
    ALL_DS = VISION_DS + IMAGENET_DS

    # Try to import open_clip, if not install
    try:
        import open_clip
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "open_clip_torch", "-q"])
        import open_clip

    results = []

    for probe_name in ["clip_vitb32", "dino_vits16"]:
        print(f"\n{'='*60}")
        print(f"Probe: {probe_name}")
        print(f"{'='*60}")

        if probe_name == "clip_vitb32":
            model, clip_preprocess = build_clip_vitb32(device)
            def forward_fn(x):
                return model(x)
            feat_dim = 512
        else:
            model = build_dino_vits16(device)
            def forward_fn(x):
                return model(x)
            feat_dim = 384

        features = {}

        # Load cached or extract new features for standard datasets
        from datasets import make_loader
        import torchvision.transforms as T

        for ds in VISION_DS:
            cache_path = os.path.join(RESULTS, f"feat_{probe_name}_{ds}.npz")
            cache_path2 = os.path.join(RESULTS, f"feat_{probe_name}_{ds}_train.npz")
            if os.path.exists(cache_path):
                features[ds] = np.load(cache_path)['H']
                print(f"  [CACHE] {ds}: {features[ds].shape}")
            elif os.path.exists(cache_path2):
                features[ds] = np.load(cache_path2)['H']
                print(f"  [CACHE] {ds}: {features[ds].shape}")
            else:
                print(f"  [EXTRACT] {ds} ...", end=" ", flush=True)
                try:
                    loader = make_loader(ds, split="train", n_samples=5000, batch_size=128)
                    feats = []
                    with torch.no_grad():
                        for images, targets in loader:
                            images = images.to(device)
                            h = forward_fn(images)
                            feats.append(h.cpu().numpy())
                    H = np.concatenate(feats, axis=0).astype(np.float32)
                    np.savez_compressed(cache_path, H=H)
                    features[ds] = H
                    print(f"OK {H.shape}")
                except Exception as e:
                    print(f"SKIP ({e})")

        # ImageNet subsets
        for gname, cids in IMAGENET_GROUPS.items():
            cache_path = os.path.join(RESULTS, f"feat_{probe_name}_{gname}.npz")
            if os.path.exists(cache_path):
                features[gname] = np.load(cache_path)['H']
                print(f"  [CACHE] {gname}: {features[gname].shape}")
            else:
                print(f"  [EXTRACT] {gname} ...", end=" ", flush=True)
                try:
                    H, Y = load_imagenet_val_features(model, forward_fn, gname, cids, device=device)
                    np.savez_compressed(cache_path, H=H, labels=Y)
                    features[gname] = H
                    print(f"OK {H.shape}")
                except Exception as e:
                    print(f"SKIP ({e})")

        # Compute all pairs
        available = [ds for ds in ALL_DS if ds in features]
        n_ds = len(available)
        n_pairs = n_ds * (n_ds - 1) // 2
        print(f"\n  Computing {n_pairs} pairs from {n_ds} datasets...")

        count = 0
        for i in range(n_ds):
            for j in range(i+1, n_ds):
                ds_A, ds_B = available[i], available[j]
                count += 1
                if count % 20 == 0: print(f"    [{count}/{n_pairs}]")
                try:
                    m = compute_pair(features[ds_A], features[ds_B])
                    dA, dB = get_domain(ds_A), get_domain(ds_B)
                    pt = dA if dA == dB else f"{dA}x{dB}"
                    results.append({'probe': probe_name, 'ds_A': ds_A, 'ds_B': ds_B, 'pair_type': pt, **m})
                except Exception as e:
                    print(f"    ERROR {ds_A}x{ds_B}: {e}")

    # Save
    outpath = os.path.join(RESULTS, "exp1_clip_dino_expanded.csv")
    if results:
        with open(outpath, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(results)} pairs")
    for p in ["clip_vitb32", "dino_vits16"]:
        sub = [r for r in results if r['probe'] == p]
        if not sub: continue
        rt = np.array([r['r_merged_true'] for r in sub])
        rp = np.array([r['r_merged_pred'] for r in sub])
        rc = np.corrcoef(rt, rp)[0,1]
        ns = sum(1 for r in sub if r['is_superadditive'])
        c = np.exp(-np.log(rp/rt).mean())
        rp_cal = rp * c
        ss = ((rt-rt.mean())**2).sum()
        r2 = 1 - ((rt-rp_cal)**2).sum()/ss
        print(f"  {p}: n={len(sub)}, r={rc:.4f}, c={c:.4f}, R2_cal={r2:.4f}, super={ns}/{len(sub)} ({100*ns/len(sub):.0f}%)")
    print(f"Saved to {outpath}")
