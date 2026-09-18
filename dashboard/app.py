import json
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

# Ensure the app can find your models folder
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from models.dataset import inference_transform
from models.resnet_model import get_model

st.set_page_config(page_title="Sentinel-AI Diagnostics", layout="wide")

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

def find_checkpoint():
    """Prefer the best-validation-AUC checkpoint the server saved."""
    for candidate in [
        ROOT / "data" / "models" / "global_best.pth",
        ROOT / "data" / "models" / "global_latest.pth",
    ]:
        if candidate.exists():
            return candidate
    legacy = sorted((ROOT / "data").glob("global_model_round_*.pth"))
    return legacy[-1] if legacy else None


@st.cache_resource
def load_cached_model():
    model_path = find_checkpoint()
    if model_path is None:
        return None, None
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, model_path


@st.cache_resource
def load_thresholds():
    """Per-class cut-offs calibrated on held-out patients by evaluate_global.py.
    Training uses pos_weight to counter class imbalance, which inflates the raw
    sigmoid outputs -- a single flat 0.10 cut-off would flag nearly everything."""
    path = ROOT / "data" / "models" / "thresholds.json"
    if path.exists():
        return json.loads(path.read_text())
    return None

# UI Header
st.title("🩺 Sentinel-AI: Federated X-Ray Diagnostics")
st.markdown("Upload a patient's chest X-ray to generate a live multi-label diagnostic report using our federated Krum-defense ResNet18 model.")

# Sidebar for uploading
with st.sidebar:
    st.header("Patient Data Input")
    uploaded_file = st.file_uploader("Upload Chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Scan")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Diagnostic Engine")
        with st.status("Initializing Sentinel-AI Model...", expanded=True) as status:
            st.write("Loading federated global weights...")
            model, model_path = load_cached_model()
            if model is None:
                status.update(label="No trained model found", state="error")
                st.error("No checkpoint found. Run training, then evaluate_global.py.")
                st.stop()
            calib = load_thresholds()
            st.write(f"Loaded `{model_path.name}`")
            
            st.write("Preprocessing image tensor...")
            # Identical preprocessing to validation -- otherwise the model sees a
            # different framing at serving time than it was scored on.
            tensor = inference_transform()(image).unsqueeze(0)

            st.write("Running clinical inference...")
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.sigmoid(outputs).squeeze().tolist()
            
            status.update(label="Analysis Complete", state="complete", expanded=False)

        st.subheader("Detected Pathologies")
        cuts = (calib or {}).get("thresholds", {})
        class_auc = (calib or {}).get("auc", {})
        if calib:
            st.caption(f"Calibrated thresholds active - validation mean AUC {calib['mean_auc']:.3f}")
        else:
            st.warning("Uncalibrated: run `python evaluate_global.py` for per-class thresholds.")

        predictions = sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)

        findings_detected = False
        for disease, prob in predictions:
            cut = cuts.get(disease, 0.5)
            if prob < cut:
                continue
            findings_detected = True
            margin = (prob - cut) / max(1.0 - cut, 1e-6)  # how far past the cut-off
            auc_note = f" | class AUC {class_auc[disease]:.2f}" if disease in class_auc else ""
            st.write(f"**{disease}** - score {prob:.2f} (threshold {cut:.2f}){auc_note}")
            st.progress(min(max(margin, 0.0), 1.0))
            st.divider()

        if not findings_detected:
            st.success("No pathology exceeded its calibrated threshold.")

        with st.expander("All 14 class scores"):
            for disease, prob in predictions:
                st.write(f"{disease}: {prob:.3f}  (threshold {cuts.get(disease, 0.5):.2f})")
else:
    st.info("Awaiting patient scan. Please upload an image using the sidebar.")