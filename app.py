import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load Model Files
PROTOTXT = "C:/Users/sreevishak/Desktop/DUK/BW yt/model/colorization_deploy_v2.prototxt"
POINTS = "C:/Users/sreevishak/Desktop/DUK/BW yt/model/pts_in_hull.npy"
MODEL = "C:/Users/sreevishak/Desktop/DUK/BW yt/model/colorization_release_v2.caffemodel"

# Load the Model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Streamlit Interface
st.title("Black-and-White Image Colorizer")
st.write("Upload a black-and-white image to colorize it!")

uploaded_file = st.file_uploader("Choose a black-and-white image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the original image
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Original Black-and-White Image", use_column_width=True)

    # Convert to OpenCV format
    image = np.array(original_image)
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Colorize
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Display the colorized image
    st.image(colorized, caption="Colorized Image", use_column_width=True)
