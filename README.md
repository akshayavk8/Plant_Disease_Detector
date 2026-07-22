# 🌿 AI-Powered Plant Disease Detection & Treatment Recommendation

An end-to-end deep learning application that detects plant diseases from leaf images and generates farmer-friendly treatment recommendations using a Hugging Face LLM.

**🔗 Live App:** [Streamlit Community Cloud](https://plantdiseasedetector-5bfzmbvkohwgmdgtxbmjn3.streamlit.app/)

## What it does

1. Upload a photo of a plant leaf
2. A trained EfficientNetB0 model predicts the disease (or confirms the plant is healthy)
3. The prediction confidence is displayed
4. The disease name is sent to a Hugging Face LLM (Qwen2.5-7B-Instruct)
5. The LLM generates a description, causes, treatment, and prevention tips
6. Results are displayed in a simple, farmer-friendly format

## Models

| Model | Test Accuracy | Notes |
|---|---|---|
| Baseline CNN | 82% | Built from scratch (Conv2D/MaxPooling/Dropout/Dense) |
| **EfficientNetB0** (deployed) | **95%** | Transfer learning, ImageNet pretrained, frozen backbone |

Trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (38 classes, ~88K images), re-split into 70% train / 15% validation / 15% test.

## Repository Contents

```
- Final_Project_Notebook.ipynb                      # Full training & evaluation notebook
- Plant_Disease_Detection_Project_Documentation     # Project Documentation
- app.py                                            # Streamlit application
- class_names.json                                  # Class index (disease name mapping)
- confusion_matrix_effnet.png                       # Evaluation visualization
- effnet_plant_disease.h5                           # Trained EfficientNetB0 model (~20MB)
- requirements.txt                                  # Python dependencies
```

## Running locally

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` with your Hugging Face token:

```toml
HF_TOKEN = "hf_your_token_here"
```

Then run:

```bash
streamlit run app.py
```

## Tech Stack

Computer Vision · Deep Learning (CNN, Transfer Learning) · TensorFlow/Keras · EfficientNet · Hugging Face Transformers · Streamlit · Python

## Full Documentation

See the project documentation (`Plant_Disease_Detection_Project_Documentation.docx`) for complete methodology, dataset details, per-class evaluation metrics, and limitations.

## Disclaimer

AI-generated treatment recommendations are for guidance only. For severe or persistent plant health issues, consult a local agricultural expert.

---
 
## 👤 Author
 
**Akshayaa V. Kumar**
Marine Biologist & Data Science Practitioner
HCL GUVI — Data Science with ML & AI Certification
 
[GitHub](https://github.com/akshayavk8) · [LinkedIn](https://linkedin.com/in/akshayavinodkumar)
 
