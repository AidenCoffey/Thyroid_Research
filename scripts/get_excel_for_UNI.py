#!/usr/bin/env python3
"""
run_inference_and_export_uni.py

— Given a directory of patient folders (each containing raw images), load a fine‑tuned
  UNI‑based classifier, run inference on every image, and export an Excel file with TWO sheets:
    Sheet “PerImage”:
      • patient_id
      • image_path
      • prob_cancer    (softmax probability that the image is cancerous)
      • prob_non       (softmax probability that the image is non‑cancerous)
      • predicted      (0=cancerous, 1=non‑cancerous)
      • confidence     (the probability corresponding to the predicted class)
      – plus an “Overall” row after each patient’s block.

    Sheet “PatientSummary” (one row per patient):
      • patient_id
      • mean_prob_cancer
      • mean_prob_non
      • overall_predicted  (0 or 1)
      • overall_confidence (based on mean_prob_cancer vs. mean_prob_non)
"""

import os
from PIL import Image
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# ----------------------------
#  CONFIGURE THESE THREE PATHS
# ----------------------------
# 1) Root directory where each subfolder is a patient (e.g. "TZ001", "TZ002", etc.).
#    Inside each patient folder are raw images (e.g. .jpg, .png, .tif, etc).
TEST_ROOT     = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING"

# 2) Path to your saved UNI‑based classifier checkpoint.
CHECKPOINT_PATH = "/home/iambrink/NOH_Thyroid_Cancer_Data/MODELS/best_uni_model.pth"

# 3) Path where the final Excel file will be written.
OUTPUT_XLSX   = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING/inference_results_uni.xlsx"

# ----------------------------
#  FIXED PARAMETERS
# ----------------------------
UNI_MODEL_NAME = "hf-hub:MahmoodLab/UNI"
NUM_CLASSES    = 2
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def is_image_file(fname: str) -> bool:
    ext = os.path.splitext(fname.lower())[1]
    return ext in IMG_EXTS


class UNIClassifier(nn.Module):
    """
    Wraps a fine‑tuned UNI VisionTransformer so that its output logits are averaged
    over patches (if necessary) to produce a [B, num_classes] tensor.
    """
    def __init__(self, uni_model: nn.Module, num_classes: int = 2):
        super(UNIClassifier, self).__init__()
        self.backbone = uni_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        outputs = self.backbone(x)
        # UNI can return either [B, num_patches, num_classes] or [B, num_classes].
        if outputs.ndim == 3:
            logits = outputs.mean(dim=1)  # average over patches → [B, num_classes]
        else:
            logits = outputs  # already [B, num_classes]
        return logits


def build_transform():
    """
    Instantiate the timm-based transforms that match the UNI model’s expected input:
      • resolve_data_config to get resize & normalize params
      • create_transform to assemble the transforms (with is_training=False for inference)
    """
    config = resolve_data_config({}, model=UNI_MODEL_NAME)
    return create_transform(**config, is_training=False)


def load_uni_model(checkpoint_path: str) -> nn.Module:
    """
    1. Instantiate the bare UNI model (no pretrained weights, num_classes=2).
    2. Load the saved state_dict from checkpoint_path.
    3. Wrap it in UNIClassifier to handle patch averaging.
    4. Return the model in eval() mode on DEVICE.
    """
    # 1) Load a fresh UNI ViT with the correct head for 2 classes
    base_uni = timm.create_model(
        UNI_MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    # 2) Load the fine‑tuned weights
    state = torch.load(checkpoint_path, map_location=DEVICE)
    base_uni.load_state_dict(state)
    base_uni.to(DEVICE).eval()

    # 3) Wrap it so forward() does patch averaging if needed
    model = UNIClassifier(base_uni, num_classes=NUM_CLASSES).to(DEVICE)
    model.eval()
    return model


def run_inference_on_folder(model: nn.Module, transform, test_root: str):
    """
    Walk through each patient subfolder under test_root. For each image:
      • Open with PIL, apply timm‐based transform → tensor [3,H,W].
      • Unsqueeze → [1,3,H,W], move to DEVICE.
      • Forward through the UN IClassifier → raw logits [1,2].
      • Softmax → probabilities [1,2].
      • Record prob_cancer = probs[0], prob_non = probs[1], predicted, and confidence.

    Returns:
      records: list of dicts {patient_id, image_path, prob_cancer, prob_non, predicted, confidence}
      per_patient_probs: dict mapping patient_id → list of (prob_cancer, prob_non).
    """
    records = []
    per_patient_probs = {}  # patient_id → list of (prob_cancer, prob_non)

    all_entries = sorted(os.listdir(test_root))
    patient_dirs = [d for d in all_entries if os.path.isdir(os.path.join(test_root, d))]

    for pid in tqdm(patient_dirs, desc="Patients"):
        folder_path = os.path.join(test_root, pid)
        image_files = sorted(os.listdir(folder_path))
        per_patient_probs[pid] = []

        for fname in image_files:
            if not is_image_file(fname):
                continue

            full_path = os.path.join(folder_path, fname)
            img = Image.open(full_path).convert("RGB")

            # Apply the timm-based transform → tensor [3,H,W]
            tensor = transform(img).unsqueeze(0).to(DEVICE)  # [1,3,H,W]

            with torch.no_grad():
                logits = model(tensor)             # [1,2]
                probs = torch.softmax(logits, dim=1).squeeze(0)  # [2]

            prob0 = float(probs[0].cpu())  # cancerous
            prob1 = float(probs[1].cpu())  # non‑cancerous
            pred = int(probs.argmax().cpu())  # 0 or 1
            confidence = float(probs[pred].cpu())

            records.append({
                "patient_id":  pid,
                "image_path":  full_path,
                "prob_cancer": prob0,
                "prob_non":    prob1,
                "predicted":   pred,
                "confidence":  confidence
            })
            per_patient_probs[pid].append((prob0, prob1))

    return records, per_patient_probs


def assemble_and_write_two_sheets(records, per_patient_probs, output_path: str):
    """
    1) Build the “PerImage” sheet: image-level rows + Overall row after each patient.
    2) Build a “PatientSummary” sheet: one row per patient with aggregated metrics.
    3) Write both sheets into a single Excel file.
    """
    # Convert records → DataFrame and sort by patient_id & image_path
    df = pd.DataFrame.from_records(records)
    df.sort_values(["patient_id", "image_path"], inplace=True)

    # Build PerImage rows (including “Overall”)
    per_image_rows = []
    for pid, group in df.groupby("patient_id"):
        # 1.a) Append each image-level row
        for _, r in group.iterrows():
            per_image_rows.append({
                "patient_id":  pid,
                "image_path":  r["image_path"],
                "prob_cancer": r["prob_cancer"],
                "prob_non":    r["prob_non"],
                "predicted":   int(r["predicted"]),
                "confidence":  r["confidence"]
            })

        # 1.b) Compute per-patient means
        probs_list = per_patient_probs[pid]  # list of (prob_cancer, prob_non)
        mean_p0 = float(sum(p[0] for p in probs_list) / len(probs_list))
        mean_p1 = float(sum(p[1] for p in probs_list) / len(probs_list))
        overall_pred = 0 if (mean_p0 >= 0.5) else 1
        overall_confidence = mean_p0 if overall_pred == 0 else mean_p1

        per_image_rows.append({
            "patient_id":  pid,
            "image_path":  "Overall",
            "prob_cancer": mean_p0,
            "prob_non":    mean_p1,
            "predicted":   overall_pred,
            "confidence":  overall_confidence
        })

    per_image_df = pd.DataFrame.from_records(
        per_image_rows,
        columns=["patient_id", "image_path", "prob_cancer", "prob_non", "predicted", "confidence"]
    )

    # Build PatientSummary rows (one per patient)
    summary_rows = []
    for pid, probs_list in per_patient_probs.items():
        mean_p0 = float(sum(p[0] for p in probs_list) / len(probs_list))
        mean_p1 = float(sum(p[1] for p in probs_list) / len(probs_list))
        overall_pred = 0 if (mean_p0 >= 0.5) else 1
        overall_confidence = mean_p0 if overall_pred == 0 else mean_p1

        summary_rows.append({
            "patient_id":          pid,
            "mean_prob_cancer":    mean_p0,
            "mean_prob_non":       mean_p1,
            "overall_predicted":   overall_pred,
            "overall_confidence":  overall_confidence
        })

    summary_df = pd.DataFrame.from_records(
        summary_rows,
        columns=["patient_id", "mean_prob_cancer", "mean_prob_non", "overall_predicted", "overall_confidence"]
    )
    summary_df.sort_values("patient_id", inplace=True)

    # Write both sheets into one Excel workbook
    with pd.ExcelWriter(output_path) as writer:
        per_image_df.to_excel(writer, sheet_name="PerImage", index=False)
        summary_df.to_excel(writer, sheet_name="PatientSummary", index=False)

    print(f"→ Wrote two-sheet Excel to {output_path}")


def main():
    # 1) Build the timm-based inference transform for UNI
    transform = build_transform()

    # 2) Load the fine‑tuned UNI classifier checkpoint
    model = load_uni_model(CHECKPOINT_PATH)

    # 3) Run inference over all patients/images
    records, per_patient_probs = run_inference_on_folder(model, transform, TEST_ROOT)

    # 4) Assemble and write both sheets to Excel
    assemble_and_write_two_sheets(records, per_patient_probs, OUTPUT_XLSX)


if __name__ == "__main__":
    main()
