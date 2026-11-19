import json
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
from torchvision import models, transforms

CONDITION_INFO = {
    "nv": {
        "name": "Melanocytic Nevi (Common Mole)",
        "about": "A benign (non-cancerous) cluster of pigment-producing cells (melanocytes) on the skin. Most people develop several moles over time, typically appearing as small, evenly colored spots or bumps.",
        "causes": "Driven by localized clustering of melanocytes; influenced by genetics and cumulative ultraviolet (UV) sunlight exposure.",
    },
    "mel": {
        "name": "Melanoma",
        "about": "A malignant (cancerous) skin tumor originating from melanocytes. It is less common than some other skin cancers but requires early medical attention because it can spread if untreated.",
        "causes": "Associated with DNA damage in melanocytes—strongly linked to intense or intermittent UV radiation exposure, fair skin/light eyes, family history, or having many/unusual moles.",
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesions",
        "about": "Non-cancerous epidermal growths such as seborrheic keratoses, solar lentigines, or lichen planus-like keratoses. They can vary in texture and color and sometimes visually resemble other lesions.",
        "causes": "Commonly linked to skin aging and cumulative sun exposure rather than infection; generally harmless on their own.",
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "about": "A common form of skin cancer that starts in the basal cells of the epidermis. It usually grows slowly and tends to stay localized, though prompt evaluation is important.",
        "causes": "Primarily associated with long-term or repeated exposure to ultraviolet (UV) radiation from sunlight or tanning beds.",
    },
    "akiec": {
        "name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "about": "Rough, scaly patches or early/intraepithelial lesions on sun-exposed skin that can sometimes be precancerous or early non-invasive carcinoma if left unchecked.",
        "causes": "Caused mainly by chronic UV light exposure over years, most often on chronically sun-exposed areas like the face, ears, forearms, or hands.",
    },
    "vasc": {
        "name": "Vascular Lesions",
        "about": "Benign or vascular growths involving blood or lymph vessels (such as angiomas, angiokeratomas, or pyogenic granulomas), often looking reddish, pink, or purplish.",
        "causes": "Arise from abnormal or clustered small blood/lymph vessels due to developmental factors, aging, hormonal shifts, or minor local irritation.",
    },
    "df": {
        "name": "Dermatofibroma",
        "about": "A small, firm, usually benign fibrous bump in the skin that often feels like a tiny hard pebble beneath the surface.",
        "causes": "Frequently appears after minor skin trauma—such as an insect bite, thorn prick, or small cut—as part of a localized fibrous tissue response.",
    },
}

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="wide")
st.title("Skin Lesion / Condition Classifier (Prototype)")
st.warning("Academic / educational prototype trained on HAM10000 (MobileNetV2). For demonstration and screening support only—not a medical diagnosis.")

@st.cache_resource
def load_model():
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    state_dict = torch.load("skin_mobilenetv2.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model, class_names = load_model()

with st.expander("About This Prototype & Dataset"):
    st.write(
        "This app uses a transfer-learning **MobileNetV2** model fine-tuned on the **HAM10000** dermatoscopic skin lesion dataset "
        "across 7 classes (`nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`)."
    )

uploaded_file = st.file_uploader("Upload a skin lesion image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]

    top3_probs, top3_indices = torch.topk(probs, k=min(3, len(class_names)))
    top_idx = int(top3_indices[0].item())
    code = class_names[top_idx]
    confidence = float(top3_probs[0].item()) * 100
    info = CONDITION_INFO.get(
        code,
        {
            "name": code,
            "about": "Skin lesion category predicted by the model.",
            "causes": "Refer to dermatological literature for condition-specific causes.",
        },
    )

    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

    with col2:
        st.subheader(f"Prediction: {code.upper()} — {info['name']}")
        st.metric("Top Confidence", f"{confidence:.2f}%")
        st.markdown("**What It Is:**")
        st.write(info["about"])
        st.markdown("**General Causes / Risk Factors:**")
        st.write(info["causes"])

        st.markdown("**Top 3 Class Probabilities:**")
        for p, idx in zip(top3_probs, top3_indices):
            c_code = class_names[int(idx.item())]
            c_pct = float(p.item()) * 100
            st.write(f"- `{c_code}` ({CONDITION_INFO.get(c_code, {}).get('name', c_code)}): **{c_pct:.2f}%**")