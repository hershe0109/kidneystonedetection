# Kidney Disease Detection with MobileNetV2

This repository contains a deep learning model based on MobileNetV2 for kidney stone detection using ultrasound images. It also provides a Streamlit app for real-time image classification.

## Contents
- **`streamlit_app/`**: Contains the `app.py` Streamlit application and supporting files for deployment.
- **`notebooks/`**: Includes Jupyter notebooks for experimentation with CNN architectures, such as batch normalization and residual connections.
  
## Usage
### Run the Streamlit App
1. Clone the repository: 
   ```bash
   git clone https://github.com/yourusername/Kidney_Stone_Detection.git
   ```
2. Navigate to the `streamlit_app/` directory:
   ```bash
   cd Kidney_Stone_Detection/streamlit_app
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Ngrok Setup
To make the Streamlit app accessible over the web using Ngrok:
```bash
ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
nohup streamlit run app.py &
ngrok http 8501
```

## File Descriptions
- **`app.py`**: The main application for uploading ultrasound images and classifying them using a pre-trained MobileNetV2 model.
- **`less.ipynb`**: A notebook version for setting up and testing the Streamlit app, along with Ngrok integration.
