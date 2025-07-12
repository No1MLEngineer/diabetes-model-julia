#@title All code in one cell
# Create the directory structure
!mkdir -p image_classifier/data

# Create the requirements.txt file
with open('image_classifier/requirements.txt', 'w') as f:
    f.write("""
streamlit
tensorflow
numpy
Pillow
""")

# Create the model.py file
with open('image_classifier/model.py', 'w') as f:
    f.write("""
import tensorflow as tf

def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
""")

# Create the train.py file
with open('image_classifier/train.py', 'w') as f:
    f.write("""
import tensorflow as tf
from model import create_model
import os

# Define constants
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = 'image_classifier/data'
MODEL_SAVE_PATH = 'image_classifier/image_classifier_model.h5'

def train_model():
    # Download and extract the dataset
    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=dataset_url, extract=True)

    base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # Create the data generators
    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    validation_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=validation_dir,
                                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                                  class_mode='binary')
    # Create the model
    model = create_model((IMG_HEIGHT, IMG_WIDTH, 3), 1)

    # Train the model
    model.fit(
        train_data_gen,
        steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=val_data_gen.samples // BATCH_SIZE
    )

    # Save the model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
""")

# Create the app.py file
with open('image_classifier/app.py', 'w') as f:
    f.write("""
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
MODEL_PATH = 'image_classifier/image_classifier_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except (IOError, ImportError):
    st.error("Error: Model not found. Please train the model first by running `train.py`.")
    st.stop()


# Define the class names
CLASS_NAMES = ['cat', 'dog']


def predict(image):
    # Preprocess the image
    img_array = np.array(image.resize((150, 150)))
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)
    score = predictions[0]

    return f"This image is {100 * (1 - score[0]):.2f}% cat and {100 * score[0]:.2f}% dog."


# Streamlit app
st.title("Cat vs. Dog Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    label = predict(image)

    st.write(label)
""")

# Install the dependencies
!pip install -r image_classifier/requirements.txt

# Run the training script
!python image_classifier/train.py

# Run the Streamlit app
!streamlit run image_classifier/app.py &>/dev/null&

# Expose the Streamlit app using ngrok
!pip install pyngrok
from pyngrok import ngrok
print(f"Streamlit app is running at: {ngrok.connect(8501)}")
