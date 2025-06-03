#!/usr/bin/env python3
"""
run_inference_and_export_vit.py

— Given a directory of patient folders (each containing raw images), load a pretrained ViT‐based classifier,
  run inference on every image, and export an Excel file with TWO sheets:
    Sheet “PerImage”:
      • patient_id
      • image_path
      • prob_cancer    (softmax probability that the image is cancerous)
      • prob_non       (softmax probability that the image is non‐cancerous)
      • predicted      (0=cancerous, 1=non‐cancerous)
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
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTFeatureExtractor

# ----------------------------
#  CONFIGURE THESE THREE PATHS
# ----------------------------
# 1) Root directory where each subfolder is a patient (e.g. "TZ001", "TZ002", etc).
#    Inside each patient folder are raw images (e.g. .jpg, .png, .tif, etc).
TEST_ROOT     = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING"
# 2) Path to your saved ViT checkpoint.
MODEL_PATH    = "/home/iambrink/NOH_Thyroid_Cancer_Data/MODELS/best_ViT_AUC_Model.pth"
# 3) Path where the final Excel file will be written.
OUTPUT_XLSX   = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING/inference_results_vit.xlsx"

# ----------------------------
#  FIXED PARAMETERS
# ----------------------------
MODEL_NAME       = "google/vit-base-patch16-224-in21k"
NUM_CLASSES      = 2
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS         = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ----------------------------
#  BUILD TRANSFORM
# ----------------------------
def build_transform():
    """
    Recreate exactly the preprocessing used during training:
      • Resize to 224×224
      • ToTensor()
      • Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    """
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
    mean = feature_extractor.image_mean
    std  = feature_extractor.image_std

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def is_image_file(fname: str) -> bool:
    ext = os.path.splitext(fname.lower())[1]
    return ext in IMG_EXTS


def load_vit_model(model_path: str) -> nn.Module:
    """
    1. Instantiate a ViTForImageClassification with num_labels=2.
    2. Load the state_dict from the checkpoint.
    3. Move to DEVICE and set to eval().
    """
    model = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def run_inference_on_folder(model: nn.Module, transform: transforms.Compose, test_root: str):
    """
    Walk through each patient subfolder under test_root. For each image:
      • apply transform
      • forward through ViT → raw logits [1, 2]
      • softmax → probabilities [2]
      • record prob_cancer (index 0), prob_non (index 1), predicted, confidence

    Returns:
      records: list of dicts {patient_id, image_path, prob_cancer, prob_non, predicted, confidence}
      per_patient_probs: dict mapping patient_id → list of (prob_cancer, prob_non)
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
            img_t = transform(img).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

            with torch.no_grad():
                outputs = model(img_t).logits          # [1, 2]
                probs   = torch.softmax(outputs, dim=1).squeeze(0)  # [2]

            prob0 = float(probs[0].cpu())  # cancerous
            prob1 = float(probs[1].cpu())  # non‐cancerous
            pred  = int(probs.argmax().cpu())  # 0 or 1
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
    1) Build “PerImage” sheet: one row per image + “Overall” row after each patient.
    2) Build “PatientSummary” sheet: one row per patient with aggregated metrics.
    3) Write both sheets into a single Excel file.
    """
    # Convert records → DataFrame, sort by patient_id & image_path
    df = pd.DataFrame.from_records(records)
    df.sort_values(["patient_id", "image_path"], inplace=True)

    # Build PerImage rows (including “Overall”)
    per_image_rows = []
    for pid, group in df.groupby("patient_id"):
        # Append each image-level row
        for _, r in group.iterrows():
            per_image_rows.append({
                "patient_id":  pid,
                "image_path":  r["image_path"],
                "prob_cancer": r["prob_cancer"],
                "prob_non":    r["prob_non"],
                "predicted":   int(r["predicted"]),
                "confidence":  r["confidence"]
            })

        # Compute per-patient means
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

    print(f"→ Wrote two‐sheet Excel to {output_path}")


def main():
    # 1) Build transform used during training
    transform = build_transform()

    # 2) Load the ViT checkpoint
    model = load_vit_model(MODEL_PATH)

    # 3) Run inference over all patients/images
    records, per_patient_probs = run_inference_on_folder(model, transform, TEST_ROOT)

    # 4) Assemble and write both sheets to Excel
    assemble_and_write_two_sheets(records, per_patient_probs, OUTPUT_XLSX)


if __name__ == "__main__":
    main()
