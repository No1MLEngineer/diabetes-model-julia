import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
MODEL_PATH = 'image_classifier_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except (IOError, ImportError):
    st.error("Error: Model not found. Please train the model first by running `train.py`.")
    st.stop()


# Define the class names
# The class names should be in the same order as the training data
# For the Intel Image Classification dataset, the classes are:
# 'buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'
# We can group them into 'urban' and 'rural'
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
URBAN_CLASSES = ['buildings', 'street']
RURAL_CLASSES = ['forest', 'glacier', 'mountain', 'sea']


def predict(image):
    """
    Predicts the class of an image.

    Args:
        image (PIL.Image.Image): The image to classify.

    Returns:
        str: The predicted class name ('urban' or 'rural').
        float: The confidence of the prediction.
    """
    # Preprocess the image
    img_array = np.array(image.resize((150, 150)))
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions[0])

    # Group the prediction into 'urban' or 'rural'
    if predicted_class_name in URBAN_CLASSES:
        return 'Urban', confidence
    else:
        return 'Rural', confidence


# Streamlit app
st.title("Urban vs. Rural Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    label, confidence = predict(image)

    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: {confidence:.2f}")
