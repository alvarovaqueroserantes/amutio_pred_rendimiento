# app/map_utils.py

import numpy as np
import folium
from pandas import DataFrame

def show_static_map(global_preds: list, ranking_df: DataFrame, parcel_coords: dict) -> str:
    """
    Renderiza un mapa folium con las parcelas coloreadas según el rendimiento predicho,
    ajustando el zoom automáticamente.
    """

    # colores normalizados
    min_pred = np.min(global_preds)
    max_pred = np.max(global_preds)

    def rendimiento_color(value):
        ratio = (value - min_pred) / (max_pred - min_pred + 1e-8)
        r = int(255 * (1 - ratio))
        g = int(180 * ratio)
        b = 0
        return f"rgb({r},{g},{b})"

    # recopilar todas las coordenadas para ajustar el fit_bounds
    all_coords = []
    for coords in parcel_coords.values():
        all_coords.extend(coords)

    mean_lat = np.mean([c[0] for c in all_coords])
    mean_lon = np.mean([c[1] for c in all_coords])

    min_lat = min(c[0] for c in all_coords)
    max_lat = max(c[0] for c in all_coords)
    min_lon = min(c[1] for c in all_coords)
    max_lon = max(c[1] for c in all_coords)

    # inicializar mapa centrado en promedio
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=13, tiles="cartodbpositron")

    # ajustar la vista para que incluya todas las parcelas
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    # añadir polígonos de las parcelas
    for parcela in ranking_df["Parcela"].tolist():
        coords = parcel_coords.get(parcela)
        if coords:
            rendimiento = ranking_df.loc[
                ranking_df["Parcela"] == parcela, "Predicción rendimiento (ton/ha)"
            ].values[0]
            color = rendimiento_color(rendimiento)

            folium.Polygon(
                locations=coords,
                color="black",
                fill=True,
                fill_opacity=0.7,
                fill_color=color,
                weight=1,
                popup=f"""
                    <b>{parcela}</b><br>
                    Rendimiento: {rendimiento:.2f} ton/ha
                """,
            ).add_to(m)

    return m._repr_html_()
