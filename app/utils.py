# app/utils.py

import numpy as np

# Codificador de variedades
VARIEDAD_ENCODER = {
    "Agria": 0,
    "Monalisa": 1,
    "Spunta": 2
}

def encode_variedad(variedad: str) -> int:
    """
    Codifica la variedad de patata en número entero.
    """
    return VARIEDAD_ENCODER.get(variedad, -1)

def build_feature_vector(datos_p, mes_cosecha: int) -> np.ndarray:
    """
    Construye el vector de entrada para el modelo de stacking
    a partir del dataframe de la parcela.
    """
    variedad_code = encode_variedad(datos_p["variedad"].iloc[-1])
    tam_parcela_ha = datos_p["tam_parcela_ha"].iloc[-1]
    temp_media = datos_p["temp_media"].mean()
    temp_std = datos_p["temp_media"].std()
    lluvia_total = datos_p["lluvia"].sum()
    fertilizante_total = datos_p["fertilizante"].sum()
    fertilizante_medio = datos_p["fertilizante"].mean()
    riego_total = datos_p["riego"].sum()
    riego_medio = datos_p["riego"].mean()

    return np.array([[variedad_code, tam_parcela_ha, temp_media, temp_std,
                      lluvia_total, fertilizante_total, fertilizante_medio,
                      riego_total, riego_medio, mes_cosecha]])

def normalize_series(series: np.ndarray) -> np.ndarray:
    """
    Normaliza series multivariadas en escala 0-1 para LSTM
    """
    return (series - series.min(axis=0)) / (series.max(axis=0) - series.min(axis=0) + 1e-8)

