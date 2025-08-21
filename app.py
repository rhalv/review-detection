import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Nama model di Hugging Face
MODEL_NAME = "rhalv/bert-review-detection"  

# Load tokenizer & model langsung dari HF Hub
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

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

