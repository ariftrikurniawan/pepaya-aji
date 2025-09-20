# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# ========== CONFIG ==========
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "mobilenetv2_pepaya.keras")  # pastikan file model ada
ALLOWED = {"png", "jpg", "jpeg"}

st.set_page_config(page_title="Pepaya Classifier", layout="centered")

# ========== CSS CUSTOM (ambil dari index.html Anda) ==========
st.markdown("""
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; background: linear-gradient(135deg, #f9f9f9, #d2fbd2); }
.container {
    background: #fff;
    padding: 30px 25px;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    text-align: center;
    max-width: 400px;
    margin: auto;
}
h1 { font-size: 1.8rem; margin-bottom: 20px; color: #2c3e50; }
.stButton>button {
    margin: 8px;
    padding: 12px 20px;
    font-size: 15px;
    font-weight: 600;
    border: none;
    border-radius: 30px;
    cursor: pointer;
    background: linear-gradient(135deg, #28a745, #4cd964);
    color: white;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}
</style>
""", unsafe_allow_html=True)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

try:
    model = load_model(MODEL_PATH)
except Exception:
    st.error("❌ Gagal memuat model. Pastikan file `mobilenetv2_pepaya.keras` ada.")
    st.stop()

# label
num_classes = model.output_shape[-1] if hasattr(model, "output_shape") else None
all_labels = ["matang", "mentah", "setengah"]
class_labels = all_labels[:num_classes] if num_classes else all_labels

# ========== UTILS ==========
def predict_image_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)
    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    label = class_labels[idx]
    conf = float(np.max(preds))
    return label, conf

# ========== UI ==========
with st.container():
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.markdown("<h1>🍈 Pepaya Maturity Classifier</h1>", unsafe_allow_html=True)

    camera_func = getattr(st, "camera_input", None)
    cam_file = camera_func("📷 Ambil Foto") if camera_func else None
    uploaded_file = st.file_uploader("🖼 Pilih dari Galeri", type=list(ALLOWED))

    file = cam_file if cam_file is not None else uploaded_file

    if file:
        try:
            file_bytes = file.getvalue()
        except Exception:
            file_bytes = file.read()

        st.image(file_bytes, caption="Preview", use_column_width=True)

        if st.button("🔍 Prediksi"):
            with st.spinner("⏳ Memproses..."):
                try:
                    label, conf = predict_image_bytes(file_bytes)
                    st.success(f"✅ Hasil: {label.upper()} ({conf*100:.2f}%)")
                except Exception as e:
                    st.error(f"❌ Error prediksi: {e}")
    else:
        st.info("Pilih atau ambil foto untuk memulai prediksi.")
    st.markdown("</div>", unsafe_allow_html=True)
