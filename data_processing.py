import numpy as np
import pandas as pd

def get_features_stack(datos_p):
    """
    Extrae el vector de features para el modelo stack
    """
    mes_cosecha = datos_p["fecha"].max().month
    variedad = datos_p["variedad"].iloc[-1]
    variedad_code = {"Agria": 0, "Monalisa": 1, "Spunta": 2}[variedad]
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

def get_lstm_series(datos_p, ciclo_weeks=16):
    """
    Prepara la secuencia temporal normalizada para el LSTM
    """
    week_series = datos_p[["temp_media", "lluvia", "fertilizante", "riego"]].values
    if len(week_series) < ciclo_weeks:
        pad = np.tile(week_series[-1], (ciclo_weeks - len(week_series), 1))
        week_series = np.vstack([week_series, pad])
    else:
        week_series = week_series[-ciclo_weeks:]

    week_series_norm = (week_series - week_series.min(axis=0)) / (
        week_series.max(axis=0) - week_series.min(axis=0) + 1e-8
    )
    return week_series_norm.reshape((1, ciclo_weeks, 4))
