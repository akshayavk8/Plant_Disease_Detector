import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import InferenceClient

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

# ---------------------------
# Load model, class names, HF client (cached so they load once)
# ---------------------------
@st.cache_resource
def load_trained_model():
    return load_model("effnet_plant_disease.h5")

@st.cache_resource
def load_class_names():
    with open("class_names.json", "r") as f:
        idx_to_class = json.load(f)
    # JSON keys are strings, convert back to int
    return {int(k): v for k, v in idx_to_class.items()}

@st.cache_resource
def get_hf_client():
    return InferenceClient(token=st.secrets["HF_TOKEN"])

model = load_trained_model()
idx_to_class = load_class_names()
client = get_hf_client()

IMG_SIZE = (224, 224)

# ---------------------------
# Helper functions
# ---------------------------
def predict_disease(image: Image.Image):
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # EfficientNet preprocessing
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    disease_name = idx_to_class[pred_idx]
    return disease_name, confidence

def get_treatment_recommendation(disease_name: str):
    clean_name = disease_name.replace("___", " - ").replace("_", " ")

    prompt = f"""Disease: {clean_name}

Provide the following in a farmer-friendly way, using simple language:
1. Disease Overview (1-2 sentences)
2. Symptoms
3. Causes
4. Recommended Treatment
5. Prevention Tips

Keep each section brief and practical."""

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate treatment recommendation right now. ({e})"

# ---------------------------
# UI
# ---------------------------
st.title("🌿 AI-Powered Plant Disease Detection")
st.write(
    "Upload a photo of a plant leaf and get an instant disease diagnosis "
    "with AI-generated treatment recommendations."
)

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing leaf image..."):
        disease_name, confidence = predict_disease(image)

    display_name = disease_name.replace("___", " - ").replace("_", " ")
    is_healthy = "healthy" in disease_name.lower()

    st.subheader("🔍 Prediction Result")
    if is_healthy:
        st.success(f"**{display_name}**")
    else:
        st.warning(f"**{display_name}**")
    st.metric("Confidence", f"{confidence * 100:.1f}%")

    if confidence < 0.5:
        st.info(
            "⚠️ Confidence is low. Try uploading a clearer, well-lit image of a single leaf "
            "for a more reliable diagnosis."
        )

    st.subheader("💊 Treatment & Care Recommendations")
    if is_healthy:
        st.write("This plant appears healthy! Keep up good care practices — proper watering, "
                 "spacing, and sunlight.")
    else:
        with st.spinner("Generating treatment recommendation..."):
            recommendation = get_treatment_recommendation(disease_name)
        st.markdown(recommendation)

    st.caption(
        "⚠️ AI-generated recommendations are for guidance only. "
        "For severe or persistent issues, consult a local agricultural expert."
    )
else:
    st.info("👆 Upload a leaf image to get started.")

st.markdown("---")
st.caption("Built with EfficientNetB0 (Transfer Learning) + Hugging Face Qwen2.5-7B-Instruct")
