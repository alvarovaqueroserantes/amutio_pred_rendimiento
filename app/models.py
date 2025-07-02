# app/models.py

import joblib
from tensorflow.keras.models import load_model

def load_stack_model(model_path="modelo_stack.pkl"):
    """
    Carga el modelo de stacking desde disco
    """
    return joblib.load(model_path)

def load_lstm_model(model_path="modelo_lstm.h5"):
    """
    Carga el modelo LSTM desde disco
    """
    return load_model(model_path, compile=False)
