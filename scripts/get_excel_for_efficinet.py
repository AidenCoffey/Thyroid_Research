import os
from PIL import Image
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# ----------------------------
#  CONFIGURE THESE THREE PATHS
# ----------------------------
TEST_ROOT   = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING"
MODEL_PATH  = "/home/iambrink/NOH_Thyroid_Cancer_Data/MODELS/EfficientNet_New_Model.pth"
OUTPUT_XLSX = "/home/iambrink/NOH_Thyroid_Cancer_Data/TAN-TESTING-RESULTS/inference_results_efficientnet.xlsx"


# ----------------------------
#  FIXED PARAMETERS
# ----------------------------
NUM_CLASSES = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ----------------------------
#  HELPER FUNCTIONS
# ----------------------------
def is_image_file(fname: str) -> bool:
    ext = os.path.splitext(fname.lower())[1]
    return ext in IMG_EXTS

def build_transform():
    """
    Preprocessing used during training:
      • Resize to 224×224
      • ToTensor()
      • Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

def load_efficientnet_model(model_path: str) -> nn.Module:
    """
    Instantiate EfficientNet-B0, replace its classifier for 2 outputs, load checkpoint, and set to eval().
    """
    model = models.efficientnet_b0(pretrained=True)
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, NUM_CLASSES)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# ----------------------------
#  INFERENCE & EXPORT
# ----------------------------
def run_inference_on_folder(model: nn.Module, transform: transforms.Compose, root: str):
    """
    Walk through each patient subfolder under root. For each image:
      • apply transform
      • forward through EfficientNet → logits [1,2]
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
        folder = os.path.join(root, pid)
        per_patient[pid] = []
        for fname in sorted(os.listdir(folder)):
            if not is_image_file(fname):
                continue
            path = os.path.join(folder, fname)
            img  = Image.open(path).convert("RGB")
            x    = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(x)
                probs  = torch.softmax(logits, dim=1).squeeze(0)

            p0 = float(probs[0].cpu())  # cancer
            p1 = float(probs[1].cpu())  # benign
            records.append({
                "patient_id":  pid,
                "image_path":  path,
                "prob_cancer": p0,
                "prob_non":    p1,
            })
            per_patient[pid].append((p0, p1))

    return records, per_patient

def assemble_and_write_two_sheets(records, per_patient, out_path: str):
    """
    Build two sheets:
      • PerImage: each image + classification string + confidence + Overall row
      • PatientSummary: one row per patient with aggregated classification string + confidence
    """
    df = pd.DataFrame.from_records(records)
    df.sort_values(["patient_id", "image_path"], inplace=True)

    # --- Sheet 1: PerImage ---
    per_rows = []
    for pid, grp in df.groupby("patient_id"):
        # per-image rows
        for _, r in grp.iterrows():
            p0 = r["prob_cancer"]
            p1 = r["prob_non"]
            if p0 >= p1:
                cls, conf = "cancerous", p0
            else:
                cls, conf = "benign", p1
            per_rows.append({
                "patient_id":   pid,
                "image_path":   r["image_path"],
                "prob_cancer":  p0,
                "prob_non":     p1,
                "classification": cls,
                "confidence":   conf
            })
        # overall summary row
        probs = per_patient[pid]
        mean0 = sum(p[0] for p in probs) / len(probs)
        mean1 = sum(p[1] for p in probs) / len(probs)
        if mean0 >= mean1:
            overall_cls, overall_conf = "cancerous", mean0
        else:
            overall_cls, overall_conf = "benign", mean1
        per_rows.append({
            "patient_id":    pid,
            "image_path":    "Overall",
            "prob_cancer":   mean0,
            "prob_non":      mean1,
            "classification": overall_cls,
            "confidence":    overall_conf
        })

    per_df = pd.DataFrame.from_records(per_rows, columns=[
        "patient_id","image_path","prob_cancer","prob_non","classification","confidence"
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
            "patient_id":         pid,
            "mean_prob_cancer":   mean0,
            "mean_prob_non":      mean1,
            "overall_classification": cls,
            "overall_confidence": conf
        })

    sum_df = pd.DataFrame.from_records(sum_rows, columns=[
        "patient_id","mean_prob_cancer","mean_prob_non","overall_classification","overall_confidence"
    ])
    sum_df.sort_values("patient_id", inplace=True)

    # write both sheets
    with pd.ExcelWriter(out_path) as writer:
        per_df.to_excel(writer, sheet_name="PerImage", index=False)
        sum_df.to_excel(writer, sheet_name="PatientSummary", index=False)

    print(f"→ Wrote two‐sheet Excel with labels to {out_path}")

def main():
    transform = build_transform()
    model     = load_efficientnet_model(MODEL_PATH)
    recs, per = run_inference_on_folder(model, transform, TEST_ROOT)
    assemble_and_write_two_sheets(recs, per, OUTPUT_XLSX)

if __name__ == "__main__":
    main()