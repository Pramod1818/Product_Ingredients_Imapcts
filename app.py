import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Google Gemini Pro Vision API and get response
def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# Function to setup input image for Gemini API
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    scale_percent = 150
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_AREA)
    denoised = cv2.fastNlMeansDenoising(resized, h=30)
    return denoised

# Function to extract text from image
def extract_text(image):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    text = re.sub(r'[.:]', ',', text)
    text = text.replace("\n", " ")
    words = text.split(",")
    # print(words)
    return words

# Streamlit UI setup
st.set_page_config(page_title="Gemini Health App")
st.header("Gemini Health App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Extract text from the image
    st.write("Extracting text from the image...")
    words = extract_text(Image.fromarray(processed_image))
    st.write("Extracted Words:", words)

    # Analyze with Gemini API
    if st.button("Analyze with Gemini API"):
        image_data = input_image_setup(uploaded_file)
        gemini_prompt = f"Act as a Ingredients Analyzer expert.What is the impact of  impact of each ingredients on the human body in detail. If ingredients full name not given, then write the full names of that code ? Codes: {', '.join(words)}"
        response = get_gemini_response(gemini_prompt)
        st.subheader("Analysis Result:")
        st.write(response)

