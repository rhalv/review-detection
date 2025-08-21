import streamlit as st
import joblib
import gdown
import os
import torch

# --- Fungsi download dari Google Drive ---
def download_file(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# === GANTI DENGAN FILE ID DRIVE KAMU ===
MODEL_ID = "1X35DNiwVgVa9bE1JmKzzvLfyug5-3qhS"           # contoh: "1aBcdEfGhijkLmNoPqrSTuvWxYz"
TOKENIZER_ID = "1XSeoSA7GtMZCS3kYLNHPQx1pFY17sRFz"
LABEL_ENCODER_ID = "1EMNfLYqlzw690arl5DXHYBsraAO1iknX"

# --- Download semua file jika belum ada ---
download_file(MODEL_ID, "model.pkl")
download_file(TOKENIZER_ID, "tokenizer.pkl")
download_file(LABEL_ENCODER_ID, "label_encoder.pkl")

# --- Load semua objek ---
model = torch.load("model.pkl", map_location=torch.device("cpu"))
tokenizer = joblib.load("tokenizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --- Streamlit UI ---
st.title("Fake Review Detector (BERT)")

user_input = st.text_area("Masukkan teks ulasan:")

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Masukkan teks terlebih dahulu!")
    else:
        # Tokenisasi input
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        # Prediksi
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.argmax(dim=1).item()

        # Konversi ke label asli
        label = label_encoder.inverse_transform([pred])[0]

        st.success(f"**Hasil Prediksi:** {label}")
