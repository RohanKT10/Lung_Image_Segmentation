import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2


# Load model
model = load_model('lung_segmentation_model.keras')
model.load_weights('lung_segmentation_model.keras')

st.title("Lung Image Segmentation App")
st.write("Upload a chest X-ray image and get the lung segmentation mask.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image, convert to grayscale and resize to 256x256
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.fit(image, (256, 256))
    st.image(image, caption='Uploaded Chest X-ray', use_container_width=True)

    # Preprocess the image for prediction
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # add batch dimension
    image_array = np.expand_dims(image_array, axis=-1)  # ensure shape (1,256,256,1)

    # Perform prediction
    pred_mask = model.predict(image_array)

    # Threshold the mask to make it binary
    mask = (pred_mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255

    # Convert mask to image for display
    mask_image = Image.fromarray(mask)

    st.image(mask_image, caption='Predicted Lung Mask', use_container_width=True)

    # Overlay mask on the original image
    image_color = np.array(image.convert("RGB"))
    mask_color = np.array(mask_image.convert("RGB"))
    overlay = cv2.addWeighted(image_color, 0.7, mask_color, 0.3, 0)
    st.image(overlay, caption='Overlay', use_container_width=True)

st.markdown("---")
st.markdown("Developed by **Rohan KT** | Lung Image Segmentation")



