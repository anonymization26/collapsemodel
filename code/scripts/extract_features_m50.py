"""
extract_features_m50.py -- Extended feature extraction, expanding the dataset pool from M=14 to M~50
Run on a remote GPU server: python extract_features_m50.py

Data sources:
  - torchvision 0.11.2 built-in datasets (KMNIST, EMNIST, QMNIST, USPS, SEMEION, Caltech101)
  - MedMNIST npz files (expected under /home/data/)
  - ImageNet val fine-grained subsets (split from the existing 5 groups into ~20 smaller semantic groups)

Output: results/feat_resnet50_{name}_train.npz (H: float32, y: int32)
"""
import os, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from PIL import Image

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
DATA_DIR = "/home/data"
os.makedirs(RESULTS, exist_ok=True)

SEED = 42
N_SAMPLES = 5000   # samples per dataset
BATCH_SIZE = 128
# Force CPU: remote has PyTorch 1.10.2+cu102 but NVIDIA L20 needs CUDA 11.8+
DEVICE = torch.device("cpu")

# ============================================================
# ResNet-50 feature extractor
# ============================================================

def build_resnet50():
    import torchvision.models as models
    backbone = models.resnet50(pretrained=True)
    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.to(DEVICE)
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def extract(model, loader, max_n=N_SAMPLES):
    """Extract features + labels from a DataLoader."""
    feats, labels = [], []
    n = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(DEVICE)
            lab = batch[1] if len(batch) > 1 else torch.zeros(len(imgs), dtype=torch.long)
            h = model(imgs).cpu().numpy()
            feats.append(h)
            labels.append(lab.numpy() if isinstance(lab, torch.Tensor) else np.array(lab))
            n += len(h)
            if n >= max_n:
                break
    H = np.concatenate(feats)[:max_n].astype(np.float32)
    Y = np.concatenate(labels)[:max_n].astype(np.int32)
    return H, Y


def save_feat(name, H, Y):
    path = os.path.join(RESULTS, f"feat_resnet50_{name}_train.npz")
    np.savez_compressed(path, H=H, y=Y)
    print(f"  [SAVED] {name}: H={H.shape}, y={Y.shape} -> {path}")


# ============================================================
# Standard transform (ResNet-50 ImageNet preprocessing)
# ============================================================

import torchvision.transforms as T

TF_224 = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class _GrayToRGB:
    def __call__(self, x):
        return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

TF_GRAY_224 = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    _GrayToRGB(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============================================================
# Dataset loaders
# ============================================================

def _subsample(ds, n, seed=SEED):
    if n is not None and len(ds) > n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), n, replace=False).tolist()
        return Subset(ds, idx)
    return ds


def _make_loader(ds, n=N_SAMPLES):
    ds = _subsample(ds, n)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


# --- torchvision datasets ---

def load_torchvision(name):
    """Load torchvision dataset (available in 0.11.2)."""
    import torchvision.datasets as dsets

    if name == "kmnist":
        ds = dsets.KMNIST(DATA_DIR, train=True, transform=TF_GRAY_224, download=True)
    elif name == "emnist_letters":
        ds = dsets.EMNIST(DATA_DIR, split="letters", train=True, transform=TF_GRAY_224, download=True)
    elif name == "emnist_digits":
        ds = dsets.EMNIST(DATA_DIR, split="digits", train=True, transform=TF_GRAY_224, download=True)
    elif name == "qmnist":
        ds = dsets.QMNIST(DATA_DIR, what="train", transform=TF_GRAY_224, download=True)
    elif name == "usps":
        ds = dsets.USPS(DATA_DIR, train=True, transform=TF_GRAY_224, download=True)
    elif name == "semeion":
        ds = dsets.SEMEION(DATA_DIR, transform=TF_GRAY_224, download=True)
    elif name == "caltech101":
        # Caltech101 data is already at /home/data/caltech101
        ds = dsets.Caltech101(DATA_DIR, transform=TF_224, download=False)
    elif name == "eurosat":
        # EuroSAT: RGB satellite images (already downloaded)
        try:
            ds = dsets.EuroSAT(DATA_DIR, transform=TF_224, download=True)
        except Exception:
            ds = dsets.EuroSAT(DATA_DIR, transform=TF_224, download=False)
    else:
        raise ValueError(f"Unknown torchvision dataset: {name}")
    return ds


# --- MedMNIST (from npz files) ---

class MedMNISTDataset(torch.utils.data.Dataset):
    """Load MedMNIST from npz file at /home/data/{name}.npz"""
    def __init__(self, name, transform=None):
        path = os.path.join(DATA_DIR, f"{name}.npz")
        data = np.load(path)
        # MedMNIST format: train_images (N,28,28) or (N,28,28,3), train_labels (N,1)
        self.images = data["train_images"]
        self.labels = data["train_labels"].squeeze().astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 2:
            img = Image.fromarray(img, mode="L")
        elif img.shape[-1] == 3:
            img = Image.fromarray(img, mode="RGB")
        else:
            img = Image.fromarray(img, mode="L")

        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


def load_medmnist(name):
    """Load MedMNIST dataset from npz."""
    # Determine if grayscale or RGB
    path = os.path.join(DATA_DIR, f"{name}.npz")
    data = np.load(path)
    img = data["train_images"][0]
    is_gray = (img.ndim == 2) or (img.ndim == 3 and img.shape[-1] == 1)
    tf = TF_GRAY_224 if is_gray else TF_224
    return MedMNISTDataset(name, transform=tf)


# --- ImageNet val subsets (fine-grained splits) ---

# Fine-grained ImageNet class groups (splitting the original 5 into ~20)
# Based on WordNet hierarchy: classes are sorted by synset ID
IMAGENET_FINE_GROUPS = {
    # From imagenet_animal (0-397) -> split into 8 groups:
    "inet_fish":          list(range(0, 50)),       # fish, sharks, rays
    "inet_reptile":       list(range(50, 100)),     # snakes, lizards, turtles
    "inet_arthropod":     list(range(100, 150)),    # spiders, insects, crustaceans
    "inet_terrier":       list(range(150, 200)),    # terrier dog breeds
    "inet_sporting_dog":  list(range(200, 250)),    # sporting/working dogs
    "inet_small_mammal":  list(range(250, 300)),    # small mammals, cats
    "inet_large_mammal":  list(range(300, 350)),    # large mammals, bears
    "inet_bird":          list(range(350, 398)),    # birds, primates

    # From imagenet_vehicle (404-547) -> split into 3 groups:
    "inet_vehicle_a":     list(range(404, 470)),    # land vehicles
    "inet_vehicle_b":     list(range(470, 547)),    # boats, aircraft, containers

    # From imagenet_artifact (547-700) -> split into 3 groups:
    "inet_kitchen":       list(range(547, 620)),    # kitchen items, utensils
    "inet_electronic":    list(range(620, 680)),    # electronics, instruments
    "inet_furniture":     list(range(680, 700)),    # furniture, home items

    # From imagenet_scene (700-924) -> split into 4 groups:
    "inet_sport_equip":   list(range(700, 770)),    # sports equipment
    "inet_music_weapon":  list(range(770, 830)),    # musical instruments, weapons
    "inet_apparel":       list(range(830, 890)),    # clothing, accessories
    "inet_structure":     list(range(890, 924)),    # buildings, structures

    # From imagenet_food (924-970) + remainder (970-1000) -> 2 groups:
    "inet_food_fruit":    list(range(924, 960)),    # food, fruits
    "inet_geology":       list(range(960, 1000)),   # geological, natural objects
}


def load_imagenet_val_subset(group_name, class_ids):
    """Load ImageNet val subset for given class IDs."""
    val_dir = os.path.join(DATA_DIR, "ILSVRC2012_img_val")
    gt_file = os.path.join(DATA_DIR, "ILSVRC2012_validation_ground_truth.txt")

    # Also check devkit path
    devkit_gt = os.path.join(DATA_DIR, "ILSVRC2012_devkit_t12", "data",
                             "ILSVRC2012_validation_ground_truth.txt")

    # Read ground truth labels (1-indexed -> 0-indexed)
    label_file = gt_file if os.path.exists(gt_file) else devkit_gt
    with open(label_file) as f:
        val_labels = [int(x.strip()) - 1 for x in f.readlines()]

    # Get sorted image files
    all_images = sorted([f for f in os.listdir(val_dir) if f.endswith('.JPEG')])
    assert len(all_images) == len(val_labels), \
        f"Mismatch: {len(all_images)} images vs {len(val_labels)} labels"

    # Filter by class IDs, remap labels to 0..num_classes-1
    class_set = set(class_ids)
    class_remap = {c: i for i, c in enumerate(sorted(class_set))}

    indices = []
    remapped_labels = []
    for i, lbl in enumerate(val_labels):
        if lbl in class_set:
            indices.append(i)
            remapped_labels.append(class_remap[lbl])

    print(f"    {group_name}: {len(indices)} images, {len(class_set)} classes")

    # Build dataset
    class ImageNetSubset(torch.utils.data.Dataset):
        def __init__(self, img_dir, image_names, labels, transform):
            self.img_dir = img_dir
            self.image_names = image_names
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.image_names)

        def __getitem__(self, idx):
            path = os.path.join(self.img_dir, self.image_names[idx])
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]

    subset_images = [all_images[i] for i in indices]
    ds = ImageNetSubset(val_dir, subset_images, remapped_labels, TF_224)
    return ds


# ============================================================
# Main extraction pipeline
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  M=50 Feature Extraction (ResNet-50)")
    print(f"  Device: {DEVICE}")
    print("=" * 72)

    model = build_resnet50()
    print("  ResNet-50 loaded.\n")

    # ---- Phase 1: torchvision datasets (only those already downloaded) ----
    TV_DATASETS = ["kmnist"]

    print("=== Phase 1: torchvision datasets ===")
    for name in TV_DATASETS:
        outpath = os.path.join(RESULTS, f"feat_resnet50_{name}_train.npz")
        if os.path.exists(outpath):
            print(f"  [SKIP] {name}: already exists")
            continue
        try:
            t0 = time.time()
            ds = load_torchvision(name)
            loader = _make_loader(ds)
            H, Y = extract(model, loader)
            save_feat(name, H, Y)
            print(f"    ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    # ---- Phase 2: MedMNIST datasets ----
    MED_DATASETS = ["octmnist", "organamnist", "organcmnist", "organsmnist",
                    "pneumoniamnist", "retinamnist", "breastmnist"]

    print("\n=== Phase 2: MedMNIST datasets ===")
    for name in MED_DATASETS:
        outpath = os.path.join(RESULTS, f"feat_resnet50_{name}_train.npz")
        if os.path.exists(outpath):
            print(f"  [SKIP] {name}: already exists")
            continue
        npz_path = os.path.join(DATA_DIR, f"{name}.npz")
        if not os.path.exists(npz_path):
            print(f"  [SKIP] {name}: npz not found at {npz_path}")
            continue
        try:
            t0 = time.time()
            ds = load_medmnist(name)
            loader = _make_loader(ds)
            H, Y = extract(model, loader)
            save_feat(name, H, Y)
            print(f"    ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    # ---- Phase 3: Fine-grained ImageNet subsets ----
    print("\n=== Phase 3: ImageNet fine-grained subsets ===")
    val_dir = os.path.join(DATA_DIR, "ILSVRC2012_img_val")
    if not os.path.isdir(val_dir):
        print(f"  [SKIP] ImageNet val not found at {val_dir}")
    else:
        for group_name, class_ids in IMAGENET_FINE_GROUPS.items():
            outpath = os.path.join(RESULTS, f"feat_resnet50_{group_name}_train.npz")
            if os.path.exists(outpath):
                print(f"  [SKIP] {group_name}: already exists")
                continue
            try:
                t0 = time.time()
                ds = load_imagenet_val_subset(group_name, class_ids)
                loader = _make_loader(ds, n=min(N_SAMPLES, len(ds)))
                H, Y = extract(model, loader, max_n=min(N_SAMPLES, len(ds)))
                save_feat(group_name, H, Y)
                print(f"    ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  [FAIL] {group_name}: {e}")

    # ---- Summary ----
    print("\n" + "=" * 72)
    existing = [f for f in os.listdir(RESULTS) if f.startswith("feat_resnet50_") and f.endswith(".npz")]
    print(f"  Total ResNet-50 feature files: {len(existing)}")
    # Count unique dataset names (strip feat_resnet50_ prefix and _train/_test suffix)
    names = set()
    for f in existing:
        n = f.replace("feat_resnet50_", "").replace("_train.npz", "").replace("_test.npz", "").replace(".npz", "")
        names.add(n)
    print(f"  Unique datasets: {len(names)}")
    for n in sorted(names):
        print(f"    - {n}")
    print("=" * 72)
    print("  Done.")
