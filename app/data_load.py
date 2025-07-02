# app/data_load.py

import pandas as pd
import numpy as np

def load_tracking_data(uploaded_file):
    """
    Lee el archivo CSV de seguimiento semanal y devuelve un DataFrame
    """
    df = pd.read_csv(uploaded_file, parse_dates=["fecha"])
    return df

def load_forecast_data(uploaded_forecast):
    """
    Lee el archivo CSV de predicción meteorológica y devuelve un DataFrame
    """
    df_forecast = pd.read_csv(uploaded_forecast, parse_dates=["fecha"])
    return df_forecast

def prepare_features(df, parcela, ciclo_weeks=16):
    """
    Prepara las features para el modelo LSTM
    """
    datos_p = df[df["parcela"] == parcela].sort_values("fecha")
    week_series = datos_p[["temp_media", "lluvia", "fertilizante", "riego"]].values

    if len(week_series) < ciclo_weeks:
        pad = np.tile(week_series[-1], (ciclo_weeks - len(week_series), 1))
        week_series = np.vstack([week_series, pad])
    else:
        week_series = week_series[-ciclo_weeks:]

    week_series_norm = (week_series - week_series.min(axis=0)) / (
        week_series.max(axis=0) - week_series.min(axis=0) + 1e-8
    )
    week_series_norm = week_series_norm.reshape((1, ciclo_weeks, 4))
    return week_series_norm
