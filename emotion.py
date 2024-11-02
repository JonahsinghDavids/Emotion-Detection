import streamlit as st
from PIL import Image
import torch
import numpy as np
from advanced_model import ResNetEmotion  # Import the advanced ResNet model
from preprocess import preprocess_image  # Import image preprocessing

# Load the trained ResNet model
model = ResNetEmotion(num_classes=7)
model.load_state_dict(torch.load("model_resnet.pth", map_location=torch.device('cpu')))
model.eval()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit app title
st.title("Emotion Detection from Uploaded Images - GUVI final project")

# File uploader for image files
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Convert to tensor for the model
    image_tensor = torch.tensor(preprocessed_image, dtype=torch.float32)

    # Model prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        emotion = emotion_labels[predicted.item()]

    # Display the prediction
    st.write(f"Detected Emotion: {emotion}")
