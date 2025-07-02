# app/forecast_utils.py

import numpy as np
import pandas as pd

def prepare_weekly_sequence(datos_p, ciclo_weeks=16):
    """
    Prepara la secuencia de datos normalizada para la entrada al LSTM.
    """
    week_series = datos_p[["temp_media", "lluvia", "fertilizante", "riego"]].values

    # padding si faltan semanas
    if len(week_series) < ciclo_weeks:
        pad = np.tile(week_series[-1], (ciclo_weeks - len(week_series), 1))
        week_series = np.vstack([week_series, pad])
    else:
        week_series = week_series[-ciclo_weeks:]

    week_series_norm = (week_series - week_series.min(axis=0)) / (
        week_series.max(axis=0) - week_series.min(axis=0) + 1e-8
    )
    return week_series_norm.reshape((1, ciclo_weeks, 4))

def calculate_future_scenarios(lstm_model, datos_p, df_forecast_p, ciclo_weeks=16):
    """
    Calcula predicciones futuras optimizadas usando el modelo LSTM
    y los datos de forecast.
    """
    week_series = datos_p[["temp_media", "lluvia", "fertilizante", "riego"]].values.tolist()
    future_opt = []

    fert_acum_opt = datos_p["fertilizante"].sum()
    riego_acum_opt = datos_p["riego"].sum()

    seq_opt = week_series.copy()
    pred_last_week = None

    for i in range(4):
        row = df_forecast_p.iloc[min(i, len(df_forecast_p)-1)]
        fert_acum_opt += row["fertilizante_sugerido_kg"] * 7
        riego_acum_opt += row["riego_sugerido_mm"] * 7
        semana_o = [
            row["temp_media"],
            row["lluvia"],
            fert_acum_opt / (len(seq_opt) + 1),
            riego_acum_opt / (len(seq_opt) + 1),
        ]
        seq_opt.append(semana_o)

        seq_o_np = np.array(seq_opt[-ciclo_weeks:])
        seq_o_norm = (seq_o_np - seq_o_np.min(axis=0)) / (
            seq_o_np.max(axis=0) - seq_o_np.min(axis=0) + 1e-8
        )
        pred_next = lstm_model.predict(seq_o_norm.reshape(1, ciclo_weeks, 4))[0][-1]
        future_opt.append(round(float(pred_next), 2))
        pred_last_week = pred_next

    return future_opt

def generate_recommendations(df_forecast_p):
    """
    Genera dataframe de recomendaciones para el usuario
    en base a la predicción meteorológica.
    """
    reco_data = []
    for _, row in df_forecast_p.iterrows():
        reco_data.append({
            "Fecha": row['fecha'].strftime("%d/%m/%Y"),
            "Temperatura alta": "Sí" if row["temp_media"] > 28 else "No",
            "Lluvia baja": "Sí" if row["lluvia"] < 2 else "No",
            "Riego sugerido": f"{row['riego_sugerido_mm']:.1f} mm/día",
            "Fertilizante sugerido": f"{row['fertilizante_sugerido_kg']:.2f} kg/día",
        })
    return pd.DataFrame(reco_data)
