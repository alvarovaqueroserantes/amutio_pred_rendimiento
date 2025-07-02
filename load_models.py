import joblib
from tensorflow.keras.models import load_model
import streamlit as st

@st.cache_resource
def load_stack_model():
    return joblib.load("models/modelo_stack.pkl")

@st.cache_resource
def load_lstm_model():
    return load_model("models/modelo_lstm.h5", compile=False)
