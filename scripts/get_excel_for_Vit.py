#!/usr/bin/env python3
"""
run_inference_and_export_vit_labels.py

— Given a directory of patient folders (each containing raw images), load a pretrained ViT‐based classifier,
  run inference on every image, and export an Excel file with TWO sheets:

    Sheet “PerImage”:
      • patient_id
      • image_path
      • prob_cancer    (softmax probability that the image is cancerous)
      • prob_non       (softmax probability that the image is non‐cancerous)
      • classification (string: "cancerous" or "benign")
      • confidence     (the probability corresponding to the chosen class)
      – plus an “Overall” row after each patient’s block.

    Sheet “PatientSummary” (one row per patient):
      • patient_id
      • mean_prob_cancer
      • mean_prob_non
      • overall_classification  (string: "cancerous" or "benign")
      • overall_confidence      (based on mean_prob_cancer vs. mean_prob_non)
"""

import os
from PIL import Image
import torch
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTFeatureExtractor

# ----------------------------
#  CONFIGURE THESE THREE PATHS
# ----------------------------
TEST_ROOT   = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING"
MODEL_PATH  = "/home/iambrink/NOH_Thyroid_Cancer_Data/MODELS/best_ViT_AUC_Model.pth"
OUTPUT_XLSX = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING/inference_results_vit_labels.xlsx"

# ----------------------------
#  FIXED PARAMETERS
# ----------------------------
MODEL_NAME = "google/vit-base-patch16-224-in21k"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def is_image_file(fname: str) -> bool:
    ext = os.path.splitext(fname.lower())[1]
    return ext in IMG_EXTS


def build_transform():
    """
    Recreate exactly the preprocessing used during training:
      • Resize to 224×224
      • ToTensor()
      • Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    """
    fe = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=fe.image_mean, std=fe.image_std)
    ])


def load_vit_model(model_path: str) -> ViTForImageClassification:
    """
    1. Instantiate ViTForImageClassification with num_labels=2.
    2. Load the state_dict from the checkpoint.
    3. Move to DEVICE and set to eval().
    """
    model = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels=2)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def run_inference_on_folder(model, transform, root: str):
    """
    Walk through each patient subfolder under root. For each image:
      • apply transform
      • forward through ViT → logits [1,2]
      • softmax → probabilities [2]
    Returns:
      records: list of {patient_id, image_path, prob_cancer, prob_non}
      per_patient: dict mapping patient_id → list of (prob_cancer, prob_non)
    """
    records = []
    per_patient = {}

    patients = sorted(d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d)))
    for pid in tqdm(patients, desc="Patients"):
        per_patient[pid] = []
        folder = os.path.join(root, pid)
        for fname in sorted(os.listdir(folder)):
            if not is_image_file(fname):
                continue
            path = os.path.join(folder, fname)
            img  = Image.open(path).convert("RGB")
            x    = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(x).logits        # [1,2]
                probs  = torch.softmax(logits, dim=1).squeeze(0)  # [2]
            p0 = float(probs[0].cpu())  # cancer
            p1 = float(probs[1].cpu())  # benign
            records.append({
                "patient_id":  pid,
                "image_path":  path,
                "prob_cancer": p0,
                "prob_non":    p1
            })
            per_patient[pid].append((p0, p1))

    return records, per_patient


def assemble_and_write_two_sheets(records, per_patient, out_path: str):
    """
    Build two sheets:
      • PerImage: per-image rows + classification string + confidence + Overall row
      • PatientSummary: one row per patient with overall classification string + confidence
    """
    df = pd.DataFrame.from_records(records)
    df.sort_values(["patient_id", "image_path"], inplace=True)

    # --- Sheet 1: PerImage ---
    per_rows = []
    for pid, grp in df.groupby("patient_id"):
        for _, r in grp.iterrows():
            p0, p1 = r["prob_cancer"], r["prob_non"]
            if p0 >= p1:
                cls, conf = "cancerous", p0
            else:
                cls, conf = "benign", p1
            per_rows.append({
                "patient_id":    pid,
                "image_path":    r["image_path"],
                "prob_cancer":   p0,
                "prob_non":      p1,
                "classification": cls,
                "confidence":    conf
            })
        # Overall summary row
        probs = per_patient[pid]
        mean0 = sum(p[0] for p in probs) / len(probs)
        mean1 = sum(p[1] for p in probs) / len(probs)
        if mean0 >= mean1:
            ocls, oconf = "cancerous", mean0
        else:
            ocls, oconf = "benign", mean1
        per_rows.append({
            "patient_id":    pid,
            "image_path":    "Overall",
            "prob_cancer":   mean0,
            "prob_non":      mean1,
            "classification": ocls,
            "confidence":    oconf
        })

    per_df = pd.DataFrame.from_records(per_rows, columns=[
        "patient_id", "image_path", "prob_cancer", "prob_non",
        "classification", "confidence"
    ])

    # --- Sheet 2: PatientSummary ---
    sum_rows = []
    for pid, probs in per_patient.items():
        mean0 = sum(p[0] for p in probs) / len(probs)
        mean1 = sum(p[1] for p in probs) / len(probs)
        if mean0 >= mean1:
            cls, conf = "cancerous", mean0
        else:
            cls, conf = "benign", mean1
        sum_rows.append({
            "patient_id":            pid,
            "mean_prob_cancer":      mean0,
            "mean_prob_non":         mean1,
            "overall_classification": cls,
            "overall_confidence":     conf
        })

    sum_df = pd.DataFrame.from_records(sum_rows, columns=[
        "patient_id", "mean_prob_cancer", "mean_prob_non",
        "overall_classification", "overall_confidence"
    ])
    sum_df.sort_values("patient_id", inplace=True)

    with pd.ExcelWriter(out_path) as writer:
        per_df.to_excel(writer, sheet_name="PerImage", index=False)
        sum_df.to_excel(writer, sheet_name="PatientSummary", index=False)

    print(f"→ Wrote two‐sheet Excel with labels to {out_path}")


def main():
    transform = build_transform()
    model     = load_vit_model(MODEL_PATH)
    recs, per = run_inference_on_folder(model, transform, TEST_ROOT)
    assemble_and_write_two_sheets(recs, per, OUTPUT_XLSX)


if __name__ == "__main__":
    main()
