#!/usr/bin/env python3
"""
run_inference_and_export_uni_labels.py

— Given a directory of patient folders (each containing raw images), load a fine-tuned
  UNI-based classifier, run inference on every image, and export an Excel file with TWO sheets:
    Sheet “PerImage”:
      • patient_id
      • image_path
      • prob_cancer    (softmax probability that the image is cancerous)
      • prob_non       (softmax probability that the image is non-cancerous)
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
TEST_ROOT       = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING"
CHECKPOINT_PATH = "/home/iambrink/NOH_Thyroid_Cancer_Data/MODELS/best_uni_model.pth"
OUTPUT_XLSX     = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING/inference_results_uni_labels.xlsx"

# ----------------------------
#  FIXED PARAMETERS
# ----------------------------
UNI_MODEL_NAME = "hf-hub:MahmoodLab/UNI"
NUM_CLASSES    = 2
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ----------------------------
#  HELPERS
# ----------------------------
def is_image_file(fname: str) -> bool:
    ext = os.path.splitext(fname.lower())[1]
    return ext in IMG_EXTS

class UNIClassifier(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x)
        if outputs.ndim == 3:
            return outputs.mean(dim=1)
        return outputs

def build_transform():
    config = resolve_data_config({}, model=UNI_MODEL_NAME)
    return create_transform(**config, is_training=False)

def load_uni_model(checkpoint_path: str) -> nn.Module:
    base = timm.create_model(UNI_MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    base.load_state_dict(state)
    base.to(DEVICE).eval()
    model = UNIClassifier(base).to(DEVICE).eval()
    return model

# ----------------------------
#  INFERENCE & EXPORT
# ----------------------------
def run_inference_on_folder(model: nn.Module, transform, root: str):
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
                logits = model(x)                   # [1,2]
                probs  = torch.softmax(logits, dim=1).squeeze(0)
            p0 = float(probs[0].cpu())  # cancerous
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
    df = pd.DataFrame.from_records(records)
    df.sort_values(["patient_id","image_path"], inplace=True)

    # PerImage sheet
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
        # Overall summary
        probs = per_patient[pid]
        mean0 = sum(p[0] for p in probs)/len(probs)
        mean1 = sum(p[1] for p in probs)/len(probs)
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
        "patient_id","image_path","prob_cancer","prob_non","classification","confidence"
    ])

    # PatientSummary sheet
    sum_rows = []
    for pid, probs in per_patient.items():
        mean0 = sum(p[0] for p in probs)/len(probs)
        mean1 = sum(p[1] for p in probs)/len(probs)
        if mean0 >= mean1:
            cls, conf = "cancerous", mean0
        else:
            cls, conf = "benign", mean1
        sum_rows.append({
            "patient_id":            pid,
            "mean_prob_cancer":      mean0,
            "mean_prob_non":         mean1,
            "overall_classification": cls,
            "overall_confidence":    conf
        })

    sum_df = pd.DataFrame.from_records(sum_rows, columns=[
        "patient_id","mean_prob_cancer","mean_prob_non",
        "overall_classification","overall_confidence"
    ])
    sum_df.sort_values("patient_id", inplace=True)

    with pd.ExcelWriter(out_path) as writer:
        per_df.to_excel(writer, sheet_name="PerImage", index=False)
        sum_df.to_excel(writer, sheet_name="PatientSummary", index=False)

    print(f"→ Wrote two-sheet Excel with labels to {out_path}")

def main():
    transform = build_transform()
    model     = load_uni_model(CHECKPOINT_PATH)
    recs, per = run_inference_on_folder(model, transform, TEST_ROOT)
    assemble_and_write_two_sheets(recs, per, OUTPUT_XLSX)

if __name__ == "__main__":
    main()
