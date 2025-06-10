#!/usr/bin/env python3
"""
run_inference_and_export_dino_labels.py

— Given a directory of patient folders (each containing raw images), load a pretrained DINOv2‐based classifier,
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
      • overall_confidence      (based on mean vs. comparison)
"""

import os
from PIL import Image
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm

# ----------------------------
#  CONFIGURE THESE THREE PATHS
# ----------------------------
TEST_ROOT     = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING"
MODEL_PATH    = "/home/iambrink/NOH_Thyroid_Cancer_Data/MODELS/best_Dino_Model_trial_7.pth"
OUTPUT_XLSX   = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING/inference_results_dino_labels.xlsx"

# ----------------------------
#  FIXED PARAMETERS
# ----------------------------
DINOV2_MODEL_NAME = "facebook/dinov2-base"
NUM_CLASSES       = 2
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS          = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def is_image_file(fname: str) -> bool:
    return os.path.splitext(fname.lower())[1] in IMG_EXTS


class DINOClassifier(nn.Module):
    def __init__(self, dinov2_model: AutoModel, num_classes: int = 2):
        super().__init__()
        self.feature_extractor = dinov2_model
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_extractor.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # extract [CLS] token
        with torch.no_grad():
            feat = self.feature_extractor(x).last_hidden_state[:, 0, :]  # [B, hidden]
        return self.classifier(feat)  # [B, 2]


def build_transform():
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_NAME)
    return processor


def load_dino_model(model_path: str) -> nn.Module:
    dinov2_base = AutoModel.from_pretrained(DINOV2_MODEL_NAME)
    model = DINOClassifier(dinov2_model=dinov2_base, num_classes=NUM_CLASSES)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


def run_inference_on_folder(model: nn.Module, processor: AutoImageProcessor, root: str):
    records = []
    per_patient = {}

    for pid in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        per_patient[pid] = []
        folder = os.path.join(root, pid)

        for fname in sorted(os.listdir(folder)):
            if not is_image_file(fname):
                continue

            path = os.path.join(folder, fname)
            img  = Image.open(path).convert("RGB")
            proc = processor(images=img, return_tensors="pt")
            pix  = proc["pixel_values"].to(DEVICE)

            with torch.no_grad():
                logits = model(pix)                         # [1,2]
                probs  = torch.softmax(logits, dim=1).squeeze(0)  # [2]

            p0 = float(probs[0].cpu())  # cancer
            p1 = float(probs[1].cpu())  # non
            records.append({
                "patient_id":    pid,
                "image_path":    path,
                "prob_cancer":   p0,
                "prob_non":      p1
            })
            per_patient[pid].append((p0, p1))

    return records, per_patient


def assemble_and_write_two_sheets(records, per_patient, out_path: str):
    df = pd.DataFrame(records)
    df.sort_values(["patient_id", "image_path"], inplace=True)

    # --- PerImage sheet ---
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

        # overall summary row
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
            "confidence":     oconf
        })

    per_df = pd.DataFrame.from_records(per_rows, columns=[
        "patient_id", "image_path", "prob_cancer", "prob_non", "classification", "confidence"
    ])

    # --- PatientSummary sheet ---
    sum_rows = []
    for pid, probs in per_patient.items():
        mean0 = sum(p[0] for p in probs) / len(probs)
        mean1 = sum(p[1] for p in probs) / len(probs)
        if mean0 >= mean1:
            cls, conf = "cancerous", mean0
        else:
            cls, conf = "benign", mean1
        sum_rows.append({
            "patient_id":             pid,
            "mean_prob_cancer":       mean0,
            "mean_prob_non":          mean1,
            "overall_classification": cls,
            "overall_confidence":     conf
        })

    sum_df = pd.DataFrame.from_records(sum_rows, columns=[
        "patient_id", "mean_prob_cancer", "mean_prob_non",
        "overall_classification", "overall_confidence"
    ])
    sum_df.sort_values("patient_id", inplace=True)

    # Write to Excel
    with pd.ExcelWriter(out_path) as writer:
        per_df.to_excel(writer, sheet_name="PerImage", index=False)
        sum_df.to_excel(writer, sheet_name="PatientSummary", index=False)

    print(f"→ Wrote Excel with labels to {out_path}")


def main():
    processor = build_transform()
    model     = load_dino_model(MODEL_PATH)
    recs, per = run_inference_on_folder(model, processor, TEST_ROOT)
    assemble_and_write_two_sheets(recs, per, OUTPUT_XLSX)


if __name__ == "__main__":
    main()
