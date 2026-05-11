import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Skin Condition Classifier",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Class Labels
# -----------------------------
CLASS_NAMES = {
    0: "Actinic Keratoses",
    1: "Basal Cell Carcinoma",
    2: "Benign Keratosis-like Lesions",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic Nevi",
    6: "Vascular Lesions"
}

# -----------------------------
# Device
# -----------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        7
    )

    model.load_state_dict(
        torch.load(
            "models/model.pth",
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model

model = load_model()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(image):

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score

# -----------------------------
# UI
# -----------------------------
st.title("🩺 Skin Condition Classification")
st.write(
    "Upload a skin lesion image and let the AI model predict the condition."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        with st.spinner("Analyzing image..."):

            prediction, confidence = predict_image(image)

        st.success(f"Prediction: {prediction}")

        st.info(
            f"Confidence Score: {confidence:.2%}"
        )


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "Built with PyTorch + Streamlit"
)