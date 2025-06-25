# kidneystonedetection Kidney Disease Detection with MobileNetV2
This repository contains a deep learning model based on MobileNetV2 for kidney stone detection using ultrasound images. It also provides a Streamlit app for real-time image classification.

Contents
streamlit_app/: Contains the app.py Streamlit application and supporting files for deployment.
notebooks/: Includes Jupyter notebooks for experimentation with CNN architectures, such as batch normalization and residual connections.
Usage
Run the Streamlit App
Clone the repository:
git clone https://github.com/yourusername/Kidney_Stone_Detection.git
Navigate to the streamlit_app/ directory:
cd Kidney_Stone_Detection/streamlit_app
Install the required packages:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run app.py
Ngrok Setup
To make the Streamlit app accessible over the web using Ngrok:

ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
nohup streamlit run app.py &
ngrok http 8501
File Descriptions
app.py: The main application for uploading ultrasound images and classifying them using a pre-trained MobileNetV2 model.
less.ipynb: A notebook version for setting up and testing the Streamlit app, along with Ngrok integration.
