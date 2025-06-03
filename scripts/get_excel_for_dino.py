#!/usr/bin/env python3
"""
run_inference_and_export_dino.py

— Given a directory of patient folders (each containing raw images), load a pretrained DINOv2‐based classifier,
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
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ----------------------------
#  CONFIGURE THESE THREE PATHS
# ----------------------------
# 1) Root directory where each subfolder is a patient (e.g. "TZ001", "TZ002", etc).
#    Inside each patient folder are raw images (e.g. .jpg, .png, .tif, etc).
TEST_ROOT     = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING"
# 2) Path to your saved DINOv2‐based classifier checkpoint.
MODEL_PATH    = "/home/iambrink/NOH_Thyroid_Cancer_Data/Notebooks_folder/best_Dino_Model_trial_7.pth"
# 3) Path where the final Excel file will be written.
OUTPUT_XLSX   = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING/inference_results_dino.xlsx"

# ----------------------------
#  FIXED PARAMETERS
# ----------------------------
DINOV2_MODEL_NAME = "facebook/dinov2-base"
NUM_CLASSES       = 2
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS          = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def is_image_file(fname: str) -> bool:
    ext = os.path.splitext(fname.lower())[1]
    return ext in IMG_EXTS


class DINOClassifier(nn.Module):
    """
    Matches the architecture used during training:
      • A frozen DINOv2 transformer (facebook/dinov2-base) as a feature extractor.
      • A small MLP classifier on top of the [CLS] token embedding.
    """
    def __init__(self, dinov2_model: AutoModel, num_classes: int = 2):
        super(DINOClassifier, self).__init__()
        self.feature_extractor = dinov2_model
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_extractor.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        with torch.no_grad():
            # DINOv2 returns a BaseModelOutputWithPooling: last_hidden_state shape [B, seq_len, hidden_size]
            features = self.feature_extractor(x).last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(features)  # [B, num_classes]


def build_transform():
    """
    Instantiate the DINOv2 processor. During training, each image was preprocessed by:
      processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
      processed = processor(images=img, return_tensors="pt")
      pixel_values = processed['pixel_values']  # shape [1,3,H,W]
    We replicate that here.
    """
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_NAME)
    return processor


def load_dino_model(model_path: str) -> nn.Module:
    """
    1. Load pretrained DINOv2 base (facebook/dinov2-base) as the frozen feature extractor.
    2. Wrap it in the same DINOClassifier architecture (MLP on top).
    3. Load weights from the saved checkpoint.
    4. Return the model in eval() mode on DEVICE.
    """
    # 1) Load the base DINOv2 transformer (frozen during inference).
    dinov2_base = AutoModel.from_pretrained(DINOV2_MODEL_NAME)
    # 2) Instantiate classifier wrapper.
    model = DINOClassifier(dinov2_model=dinov2_base, num_classes=NUM_CLASSES)
    # 3) Load state_dict
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    # 4) Move to device & eval
    model.to(DEVICE)
    model.eval()
    return model


def run_inference_on_folder(model: nn.Module, processor: AutoImageProcessor, test_root: str):
    """
    Walk through each patient subfolder under test_root. For each image:
      • Apply DINOv2 preprocessing via processor.
      • Forward through the classifier → raw logits [1,2].
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
            # Preprocess via DINOv2 processor
            processed = processor(images=img, return_tensors="pt")
            pixel_values = processed["pixel_values"].to(DEVICE)  # [1, 3, H, W]

            with torch.no_grad():
                logits = model(pixel_values)             # [1, 2]
                probs = torch.softmax(logits, dim=1).squeeze(0)  # [2]

            prob0 = float(probs[0].cpu())  # cancerous
            prob1 = float(probs[1].cpu())  # non‐cancerous
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

    print(f"→ Wrote two‐sheet Excel to {output_path}")


def main():
    # 1) Build DINOv2 preprocessing transform
    processor = build_transform()

    # 2) Load the pretrained DINOv2 classifier checkpoint
    model = load_dino_model(MODEL_PATH)

    # 3) Run inference over all patients/images
    records, per_patient_probs = run_inference_on_folder(model, processor, TEST_ROOT)

    # 4) Assemble and write both sheets to Excel
    assemble_and_write_two_sheets(records, per_patient_probs, OUTPUT_XLSX)


if __name__ == "__main__":
    main()
