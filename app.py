import joblib
import streamlit as st
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

st.set_page_config(page_title="Bottle Defect Detection", page_icon="🔎")

st.title("Bottle Defect Detection")
st.write(
    "Upload a bottle image to check whether it looks unusual "
    "compared with normal bottles in the training dataset."
)


@st.cache_resource
def load_detector():
    detector = joblib.load("detector.joblib")

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()

    return detector, model, weights.transforms()


uploaded_file = st.file_uploader(
    "Choose a bottle image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=350)

    if st.button("Analyze image"):
        with st.spinner("Analyzing..."):
            detector, model, preprocess = load_detector()
            tensor = preprocess(image).unsqueeze(0)

            with torch.no_grad():
                features = model(tensor).numpy()

            distance, _ = detector["neighbors"].kneighbors(features)
            score = float(distance[0, 0])
            threshold = detector["threshold"]

        if score > threshold:
            st.warning("Potential anomaly detected")
        else:
            st.success("No anomaly detected")

        st.write(f"Anomaly score: **{score:.2f}**")
        st.write(f"Detection threshold: **{threshold:.2f}**")

        st.caption(
            "This is a demonstration trained on MVTec bottle images. "
            "It is not intended for real-world quality inspection."
        )
        