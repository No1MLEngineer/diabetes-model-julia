# Image Classifier: Urban vs. Rural

This repository contains a deep learning-based image classifier designed to distinguish between urban and rural landscapes. The project is built using TensorFlow for the model, Streamlit for the interactive web application, and Docker for containerization. It also includes a CI/CD pipeline using GitHub Actions for automated deployment to Google Cloud Platform (GCP).

## Key Features

*   **Deep Learning Model:** A Convolutional Neural Network (CNN) built with TensorFlow and Keras for image classification.
*   **Interactive Web App:** A user-friendly web interface created with Streamlit that allows for easy image uploads and classification.
*   **Containerized Application:** A Dockerfile is provided for building a portable and reproducible container for the application.
*   **Automated Deployment:** A GitHub Actions workflow is configured for continuous integration and deployment to Google Cloud Run.
*   **Colab-Ready:** A single-file version of the code is available for easy experimentation in Google Colab.

## Getting Started

There are several ways to run this project, depending on your needs and environment.

### 1. Google Colab (Recommended for Quick-Start)

For a quick and easy way to see the project in action, you can use the provided Google Colab notebook.

1.  Open Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)
2.  Select **File > New notebook**.
3.  Copy the entire content of the `image_classifier/colab.py` file from this repository.
4.  Paste the code into a cell in your new Colab notebook.
5.  Run the cell by pressing **Shift + Enter**.

This will handle all dependencies, train the model, and launch the Streamlit application. An `ngrok` URL will be generated, allowing you to access the web app directly from your browser.

### 2. Local Development

To run the application on your local machine, you'll need Python, Docker, and the Google Cloud SDK installed.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r image_classifier/requirements.txt
    ```

3.  **Train the model:**
    *   **Note:** You will need to have a dataset of images. The training script is configured to work with the "Intel Image Classification" dataset from Kaggle. Please download and place the dataset in a `data` directory inside the `image_classifier` directory.
    ```bash
    python image_classifier/train.py
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run image_classifier/app.py
    ```

Alternatively, you can use Docker to build and run the application in a container:

```bash
docker build -t image-classifier ./image_classifier
docker run -p 8501:8501 image-classifier
```

### 3. Deployment to Google Cloud Platform

This project is configured for automated deployment to Google Cloud Run. To enable this, you will need to perform the following setup:

1.  **GCP Project Setup:**
    *   Create a new project in the [Google Cloud Console](https://console.cloud.google.com/).
    *   Enable the following APIs for your project:
        *   Cloud Build API
        *   Cloud Run API
        *   Container Registry API

2.  **Service Account:**
    *   Create a new service account in your GCP project.
    *   Grant the service account the following roles:
        *   **Cloud Run Admin**
        *   **Storage Admin**
    *   Create a JSON key for the service account and download it.

3.  **GitHub Secrets:**
    *   In your GitHub repository, navigate to **Settings > Secrets**.
    *   Add the following secrets:
        *   `GCP_PROJECT_ID`: Your Google Cloud project ID.
        *   `GCP_SA_KEY`: The entire content of the JSON service account key file you downloaded.

4.  **Trigger Deployment:**
    *   Push your code to the `main` branch. This will automatically trigger the GitHub Actions workflow, which will build the Docker image, push it to Google Container Registry, and deploy it to Cloud Run.

## Project Structure

```
.
├── .github/workflows/main.yml  # GitHub Actions workflow for CI/CD
├── image_classifier/
│   ├── app.py                  # Streamlit web application
│   ├── colab.py                # All-in-one code for Google Colab
│   ├── Dockerfile              # Dockerfile for containerization
│   ├── model.py                # CNN model definition
│   ├── requirements.txt        # Python dependencies
│   └── train.py                # Script for training the model
└── README.md                   # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request with any improvements or new features.
