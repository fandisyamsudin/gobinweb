import os
import json
import requests
import state
import streamlit as st
import tensorflow as tf
# from utils import load_and_prep_image, classes_and_models, update_logger, predict_json
from predict import predict
from image import preprocess_image
# from logger import result_logger

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "go-bin-capstone-1516532ab6ab.json" # credential service account
PROJECT = "go-bin-capstone" # ID Project
REGION = "asia-southeast1" # Region host model

st.title("Gobin Web")
st.header("Identify what's in your plastic code!!!")

@st.cache 
def make_prediction(image, model, class_names):
    image = preprocess_image(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    preds = predict(project=PROJECT,
                         region=REGION,
                         model=model,
                         instances=image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    return image, pred_class, pred_conf

plastic_class = ['1_polyethylene_PET',
 '2_high_density_polyethylene_PE-HD',
 '3_polyvinylchloride_PVC',
 '4_low_density_polyethylene_PE-LD',
 '5_polypropylene_PP',
 '6_polystyrene_PS',
 '7_other_resins',
 '8_no_plastic']

plastic_model = "fandi_efficientnet_model"

if st.checkbox("Show plastic code?"):
    st.write(f"Your model is {plastic_model}, the plastic code have 8 classes:\n", plastic_class)

uploaded_file = st.file_uploader(label="Upload your plastic code an image",
                                 type=["png", "jpeg", "jpg"])

session_state = state.get(pred_button=False)

if not uploaded_file:
    st.warning("Please upload your plastic code an image")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict image")

if pred_button:
    session_state.pred_button = True 

if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image, model=plastic_model, class_names=plastic_class)
    st.write(f"Prediction: {session_state.pred_class}, \
               Confidence: {session_state.pred_conf*10:.2f}%")