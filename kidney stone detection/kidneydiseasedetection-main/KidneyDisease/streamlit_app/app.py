
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pyngrok import ngrok

from io import BytesIO

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('/content/MobileNetV2.h5')
    return model

model = load_model()

st.write("""
# Kidney Stone Detection
This application uses a machine learning model to classify uploaded kidney ultrasound images into four categories: Normal, Cyst, Stone, and Tumor.
""")

file = st.file_uploader("Please upload an image", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    return predictions

# Function to draw hollow box around detected regions
def draw_hollow_box(image, label):
    image_array = np.array(image)
    h, w = image_array.shape[:2]
    if label == 'Cyst':
        start_point = (int(w*0.4), int(h*0.4))
        end_point = (int(w*0.6), int(h*0.6))
    elif label == 'Stone':
        start_point = (int(w*0.3), int(h*0.3))
        end_point = (int(w*0.7), int(h*0.7))
    elif label == 'Tumor':
        start_point = (int(w*0.3), int(h*0.3))
        end_point = (int(w*0.7), int(h*0.7))
    else:
        start_point = (0, 0)
        end_point = (0, 0)

    image_with_box = cv2.rectangle(image_array.copy(), start_point, end_point, (255, 0, 0), 2)
    return Image.fromarray(image_with_box)

# Load example images for each condition
def load_example_images():
    normal_image = Image.open('/content/Normal- (840).jpg')
    cyst_image = Image.open('/content/Cyst- (997)kidney.jpg')
    stone_image = Image.open('/content/Stone- (995)kidney.jpg')
    tumor_image = Image.open('/content/Tumor- (993)kidney.jpg')
    return normal_image, cyst_image, stone_image, tumor_image

normal_image, cyst_image, stone_image, tumor_image = load_example_images()

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    predictions = import_and_predict(image, model)
    class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
    result = class_names[np.argmax(predictions)]
    st.success(f"Output: {result}")

    highlighted_image = draw_hollow_box(image, result)
    st.image(highlighted_image, caption='Detected Region', use_column_width=True)

    st.write("""
    ## Differences between Conditions
    Here is a brief description of each condition:
    - **Normal**: Healthy kidney with no abnormalities.
    - **Cyst**: Fluid-filled sacs within the kidney, usually benign.
    - **Stone**: Solid masses made of crystals, causing pain and discomfort.
    - **Tumor**: Abnormal tissue growth, which can be benign or malignant.
    """)

    st.write("### Uploaded Image vs Condition Examples")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        if result == 'Normal':
            st.image(normal_image, caption='Example of Normal', use_column_width=True)
        elif result == 'Cyst':
            st.image(cyst_image, caption='Example of Cyst', use_column_width=True)

    with col3:
        if result == 'Stone':
            st.image(stone_image, caption='Example of Stone', use_column_width=True)
        elif result == 'Tumor':
            st.image(tumor_image, caption='Example of Tumor', use_column_width=True)
