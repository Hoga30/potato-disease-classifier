import streamlit as st
import torch
from PIL import Image
from torchvision import transforms


# Load the trained TorchScript model
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Rose-30/potato-disease-classifier",
    filename="potato_disease_classifier.pt"
)

model = torch.jit.load(
    model_path,
    map_location="cpu"
)

model.eval()


# Class names
classes = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Page configuration
st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔"
)


# App title
st.title("🥔 Potato Disease Classifier")

st.write(
    "Upload a potato leaf image to classify it as "
    "Healthy, Early Blight, or Late Blight."
)


# Upload image
uploaded_file = st.file_uploader(
    "Upload Potato Leaf Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Potato Leaf",
        use_container_width=True
    )

    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    # Get predicted class
    predicted_index = probabilities.argmax().item()
    predicted_class = classes[predicted_index]
    confidence = probabilities[predicted_index].item()

    # Display prediction
    st.subheader("Prediction")

    st.success(
        f"{predicted_class} "
        f"({confidence * 100:.2f}% confidence)"
    )

    # Display probabilities
    st.subheader("Class Probabilities")

    for i, class_name in enumerate(classes):
        probability = probabilities[i].item()

        st.write(
            f"{class_name}: {probability * 100:.2f}%"
        )

        st.progress(probability)