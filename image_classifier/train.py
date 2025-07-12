import tensorflow as tf
from model import create_model
import os

# Define constants
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = 'data' # This should be the directory where the dataset is stored
MODEL_SAVE_PATH = 'image_classifier_model.h5'

def train_model():
    """
    Trains the image classification model.
    """
    # Check if the dataset directory exists
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found.")
        print("Please download the Intel Image Classification dataset from Kaggle:")
        print("https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
        print(f"And place it in the '{DATASET_DIR}' directory.")
        return

    # Load the training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # Load the validation data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # Get the class names
    class_names = train_ds.class_names
    print(f"Class names: {class_names}")
    num_classes = len(class_names)

    # Create the model
    model = create_model((IMG_HEIGHT, IMG_WIDTH, 3), num_classes)

    # Train the model
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Save the model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
