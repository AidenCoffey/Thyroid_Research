import os
import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from captum.attr import Occlusion
from PIL import Image
import numpy as np

# Use a non‐interactive backend for matplotlib to avoid Qt errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------------------
# 1) Reproducibility & Device Setup
# -----------------------------------
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------
# 2) Paths & Constants
# -----------------------------------
CSV_PATH        = "/home/iambrink/NOH_Thyroid_Cancer_Data/CSV-files/Thyroid_Cancer_TAN&NOH_file.csv"
BASE_IMAGE_PATH = "/home/iambrink/NOH_Thyroid_Cancer_Data/superdata/"
CHECKPOINT_PATH = "/home/iambrink/NOH_Thyroid_Cancer_Data/MODELS/best_virchow2_5.pth"
OUTPUT_DIR      = "CAUTUM_TEST"      # Folder to save all heatmap images
MODEL_NAME      = "hf-hub:paige-ai/Virchow2"
NUM_CLASSES     = 2
BATCH_SIZE      = 1                 # Process one image at a time for Occlusion
OC_SIZE         = 8                 # Occlusion patch size (8×8)
OC_STRIDE       = 4                 # Occlusion stride
IMAGE_COLUMN    = "image_path"      # CSV column name for image paths
LABEL_COLUMN    = "Surgery diagnosis in number"

# -----------------------------------
# 3) Make Output Directory
# -----------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------
# 4) Dataset Definition
# -----------------------------------
class ThyroidDataset(Dataset):
    def __init__(self, df, base_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.base = base_path
        self.tf = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row[IMAGE_COLUMN].replace("\\", "/")
        img_path = os.path.join(self.base, rel_path)
        pil = Image.open(img_path).convert("RGB")
        img_tensor = self.tf(pil)  # shape: (C, H, W)
        label = int(row[LABEL_COLUMN])
        return img_tensor, torch.tensor(label, dtype=torch.long), img_path

# -----------------------------------
# 5) Prepare Transform & Test Split
# -----------------------------------
dummy_model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=NUM_CLASSES,
    mlp_layer=timm.layers.SwiGLUPacked,
    act_layer=torch.nn.SiLU
)
config        = resolve_data_config({}, model=dummy_model)
val_transform = create_transform(**config, is_training=False)

df = pd.read_csv(CSV_PATH).dropna(subset=[LABEL_COLUMN])
_, test_df = train_test_split(
    df,
    test_size=0.1,
    random_state=seed,
    stratify=df[LABEL_COLUMN]
)

test_ds     = ThyroidDataset(test_df, BASE_IMAGE_PATH, transform=val_transform)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# -----------------------------------
# 6) Load Virchow2 & Define Forward Function
# -----------------------------------
model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=NUM_CLASSES,
    mlp_layer=timm.layers.SwiGLUPacked,
    act_layer=torch.nn.SiLU
).to(device)

state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.eval()

def forward_func(input_tensor):
    """
    Given input_tensor of shape (B, C, H, W), return
    a tensor of shape (B,) containing the probability of class 1
    (the positive/cancer class), using the [CLS] token from Virchow2’s output.
    """
    output = model(input_tensor)         # → (B, seq_length, 2)
    cls_logits = output[:, 0, :]         # → (B, 2)
    probs = torch.softmax(cls_logits, dim=1)
    return probs[:, 1]                   # shape: (B,) = P(class 1)

# -----------------------------------
# 7) Set Up Captum Occlusion
# -----------------------------------
occlusion = Occlusion(lambda x: forward_func(x))

# -----------------------------------
# 8) Loop Over Test Set, Compute & Save Heatmaps
# -----------------------------------
for img_tensor, label_tensor, img_path in tqdm(test_loader, desc="Occlusion"):
    img_tensor = img_tensor.to(device)       # shape: (1, C, H, W)
    img_path = img_path[0]                   # string path, since batch_size=1

    original_pil = Image.open(img_path).convert("RGB")

    # 8.1) Compute the model’s base probability for class 1
    with torch.no_grad():
        orig_prob = forward_func(img_tensor).item()

    # 8.2) Compute Occlusion attributions
    attributions = occlusion.attribute(
        img_tensor,
        sliding_window_shapes=(1, OC_SIZE, OC_SIZE),
        strides=(1, OC_STRIDE, OC_STRIDE)
    )  # → shape (1, C, H, W)

    # 8.3) Collapse over channel to get a single heatmap (H, W)
    heatmap_tensor = attributions.squeeze(0).mean(dim=0)    # → (H, W)
    heatmap_np = heatmap_tensor.detach().cpu().numpy()

    # 8.4) Normalize heatmap to [0, 1]
    hm_min, hm_max = heatmap_np.min(), heatmap_np.max()
    heatmap_norm = (heatmap_np - hm_min) / (hm_max - hm_min + 1e-8)

    # 8.5) Plot side‐by‐side: original & heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_pil)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap_norm, cmap="jet", interpolation="nearest")
    axes[1].set_title(f"Occlusion Heatmap\nP(class 1)={orig_prob:.3f}")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 8.6) Save figure to CAUTUM_TEST folder
    image_id = os.path.splitext(os.path.basename(img_path))[0]
    save_filename = f"{image_id}.jpg"
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    # Use tqdm.write to log without breaking the progress bar
    tqdm.write(f"Saved heatmap for {image_id}")

print("\n→ All heatmaps have been saved in folder:", OUTPUT_DIR)
