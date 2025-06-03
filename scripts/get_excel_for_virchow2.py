#!/usr/bin/env python3
"""
run_inference_and_export.py

— Given a directory of patient folders (each containing raw images), load a pretrained Virchow2 model,
  run inference on every image, and export an Excel file that lists, per‐image:
    • patient_id
    • image_path
    • confidence   (probability of the predicted class)
    • classification (0=cancerous, 1=non‐cancerous)
  Then, immediately after each patient’s images, write one summary row where image_path == "Overall",
  with columns:
    • patient_id
    • image_path = "Overall"
    • confidence = (mean probability of the cancerous class across that patient’s images)
    • classification = 0 if (mean cancerous prob ≥ 0.5) else 1
"""

import os
import sys
from PIL import Image
import torch
import torch.nn as nn
import timm
import pandas as pd
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

# ----------------------------
#  CONFIGURE THESE THREE PATHS
# ----------------------------
# 1) Root directory where each subfolder is a patient (e.g. "TZ001", "TZ002", etc).
#    Inside each patient folder are raw images (e.g. .jpg, .png, .tif, etc).
TEST_ROOT     = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING"
# 2) Path to your saved model checkpoint (the best_virchow2_*.pth file).
MODEL_PATH    = "/home/iambrink/NOH_Thyroid_Cancer_Data/Notebooks_folder/best_virchow2_4.pth"
# 3) Path where the final Excel file will be written.
OUTPUT_XLSX   = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING/inference_results-virchow2.xlsx"

# ----------------------------
#  FIXED HYPERPARAMETERS
# ----------------------------
MODEL_NAME    = "hf-hub:paige-ai/Virchow2"
NUM_CLASSES   = 2
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 1   # we’ll do per‐image inference (B=1) for simplicity

# Allowed image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def is_image_file(fname):
    ext = os.path.splitext(fname.lower())[1]
    return ext in IMG_EXTS


def build_val_transform():
    """
    Recreate exactly the validation-time transform you used during training.
    """
    config        = resolve_data_config({}, model=MODEL_NAME)
    val_transform = create_transform(**config, is_training=False)
    return val_transform


def load_model(model_path):
    """
    1. Create a Virchow2 model with the same architecture you used,
       freeze its backbone, leave the classifier trainable (though at inference time it doesn't matter).
    2. Load the saved state_dict into it.
    3. Return the model in eval() mode on DEVICE.
    """
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,       # we will load weights from checkpoint
        num_classes=NUM_CLASSES,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU
    )

    # Freeze all params except the classifier (matches training).
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def run_inference_on_folder(model, transform, test_root):
    """
    Walk through each patient subfolder under test_root. For each image:
      • apply transform
      • forward through the model to get patch_logits → mean → final logits
      • softmax → probabilities
      • predicted = argmax, confidence = probs[predicted]

    Returns:
      records: list of dicts with per-image outputs
      per_patient_probs: dict mapping patient_id → list of (prob_cancer, prob_non) for each image
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
            img_t = transform(img).unsqueeze(0).to(DEVICE)  # [1,3,H,W]

            with torch.no_grad():
                patch_logits = model(img_t)             # [1, num_patches, 2]
                logits       = patch_logits.mean(dim=1)  # [1,2]
                probs        = torch.softmax(logits, dim=1).squeeze(0)  # [2]

            prob0 = float(probs[0].cpu())  # cancerous prob
            prob1 = float(probs[1].cpu())  # non‐cancerous prob
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


def assemble_and_write_two_sheets(records, per_patient_probs, output_path):
    """
    1) Build the “PerImage” sheet exactly as before (image-level rows + Overall row).
    2) Build a “PatientSummary” sheet: one row per patient with aggregated stats.
    3) Write both sheets into the same Excel file.
    """
    # Convert records → DataFrame and sort
    df = pd.DataFrame.from_records(records)
    df.sort_values(["patient_id", "image_path"], inplace=True)

    # Build the PerImage rows (including “Overall” after each patient)
    per_image_rows = []
    for pid, group in df.groupby("patient_id"):
        # Image-level rows
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

    # Build the PatientSummary sheet (one row per patient)
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

    # Write both sheets into one Excel file
    with pd.ExcelWriter(output_path) as writer:
        per_image_df.to_excel(writer, sheet_name="PerImage", index=False)
        summary_df.to_excel(writer, sheet_name="PatientSummary", index=False)

    print(f"→ Wrote two-sheet Excel to {output_path}")


def main():
    # 1) Build validation transform
    val_tf = build_val_transform()

    # 2) Load the pretrained checkpoint
    model = load_model(MODEL_PATH)

    # 3) Run inference across all patients/images
    records, per_patient_probs = run_inference_on_folder(model, val_tf, TEST_ROOT)

    # 4) Assemble and write both sheets
    assemble_and_write_two_sheets(records, per_patient_probs, OUTPUT_XLSX)


if __name__ == "__main__":
    main()