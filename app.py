
import streamlit as st
import numpy as np
import cv2
import json

from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Bengali OCR",
    page_icon="🔤",
    layout="centered"
)

@st.cache_resource
def load_resources():

    model = load_model("bengali_ocr_model.keras")

    classes = np.load(
        "label_classes.npy",
        allow_pickle=True
    )

    enc = LabelEncoder()
    enc.classes_ = classes

    with open(
        "bengali_map.json",
        "r",
        encoding="utf-8"
    ) as f:

        bmap = json.load(f)

    return model, enc, bmap

model, encoder, bengali_map = load_resources()

# ── Header ──────────────────────────────────────────

st.title("🔤 Bengali Handwriting Recognition")

st.markdown(
    "Upload a handwritten **Bengali character** image."
)

st.markdown("---")

# ── Upload ───────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload image (JPG / PNG / BMP)",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded image")

        st.image(
            image,
            use_column_width=True
        )

    # ── Preprocess ────────────────────────────────────

    img_array = np.array(
        image.convert("L")
    )

    img_resized = cv2.resize(
        img_array,
        (64, 64)
    )

    img_norm = img_resized / 255.0

    img_input = img_norm.reshape(
        1,
        64,
        64,
        1
    )

    # ── Predict ───────────────────────────────────────

    pred = model.predict(img_input)

    pred_idx = int(np.argmax(pred))

    confidence = float(np.max(pred)) * 100

    pred_label = encoder.inverse_transform(
        [pred_idx]
    )[0]

    pred_char = bengali_map.get(
        str(pred_label),
        "?"
    )

    with col2:

        st.subheader("Prediction")

        st.markdown(
            f"""

# {pred_char}

**Label:** {pred_label}

**{confidence:.1f}% confidence**

""",
            unsafe_allow_html=True
        )

    # ── Top 5 ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Top 5 predictions")

    top5 = np.argsort(
        pred[0]
    )[::-1][:5]

    for rank, idx in enumerate(top5, 1):

        lbl = encoder.inverse_transform(
            [idx]
        )[0]

        char = bengali_map.get(
            str(lbl),
            "?"
        )

        conf = float(pred[0][idx] * 100)

        st.markdown(
            f"**#{rank}** {char} — label {lbl}"
        )

        st.progress(float(conf) / 100)

# ── Sidebar ──────────────────────────────────────────

st.sidebar.title("About")

st.sidebar.info(
    "Bengali OCR using CNN trained on CMATERdb\n\n"
    f"Classes: 50 Bengali characters\n\n"
    "Upload a single handwritten character image to test."
)
