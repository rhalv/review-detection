import streamlit as st
import gdown
import os
import zipfile
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# --- Download file model ---
MODEL_ID = "1RuK0hMeWKCPi2t0H5Nz5JfRMJZ0Ti_uR"

if not os.path.exists("bert_model"):
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, "bert_model.zip", quiet=False)

    with zipfile.ZipFile("bert_model.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# --- Load model & tokenizer ---
tokenizer = BertTokenizer.from_pretrained("bert_model")
model = BertForSequenceClassification.from_pretrained(
    "bert_model", 
    device_map="cpu"   
)

# --- Streamlit UI ---
st.title("Fake Review Detector (BERT)")

user_input = st.text_area("Masukkan teks ulasan:")

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Masukkan teks terlebih dahulu!")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.argmax(dim=1).item()

        label = "Asli" if pred == 1 else "Palsu"  # sesuaikan label kamu
        st.success(f"**Hasil Prediksi:** {label}")
