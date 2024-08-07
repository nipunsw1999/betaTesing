import streamlit as st 
import cv2
import numpy as np
from PIL import Image

st.title("Sinhala OCR Beta Testing")

image = st.file_uploader("Choose an image")

thresh = st.select_slider("Select Threshold value", options=[i for i in range(0, 256)])

def im(img):
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    return gray

def im2gray(img, thresh):
    _, binary_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
    return binary_img  

if image is not None:
    img = Image.open(image)
    gray_image = im(img)
    binary_image = im2gray(gray_image, thresh)
    st.image(binary_image, caption="Binary Image", use_column_width=True)

def sentence_segmentation(img, thresh):
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    w, h = img.size
    sentences = []
    array2 = np.zeros((w), dtype=np.uint8)
    current_sentence = []

    for height in range(h):
        boo = False
        for width in range(w):
            array2[width] = binary_img[height, width]
            if binary_img[height, width] == 255:
                boo = True

        if boo:
            current_sentence.append(array2.copy())
        else:
            if current_sentence:
                sentences.append(np.array(current_sentence))
                current_sentence = []

    if current_sentence:
        sentences.append(np.array(current_sentence))

    return sentences

if image is not None:
    st.title("Separated Sentences")
    sentences = sentence_segmentation(img, thresh)
    number_of_sentences = len(sentences)

    for i in range(number_of_sentences):
        st.image(sentences[i], caption=f"Sentence No {i+1}", use_column_width=True)
