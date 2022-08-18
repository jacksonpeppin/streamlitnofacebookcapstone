from PIL import Image
import urllib.request
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 50
CATEGORIES = ["Human Detected", "No Human Detected"]
model = tf.keras.models.load_model("./models/64x3x1-CNN-no-tensorboard.model")


def classify_img(prepared_img):
    prediction = model.predict(prepared_img)
    return CATEGORIES[int(prediction[0][0])]


def url_to_image(img_url):
    url_response = urllib.request.urlopen(img_url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    new_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return new_img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

