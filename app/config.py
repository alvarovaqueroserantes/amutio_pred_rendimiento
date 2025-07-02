# app/config.py

PAGE_TITLE = "AMUTIO Predictive Dashboard"
PAGE_ICON = "images/logo.png"

# colores corporativos
COLOR_MEAN = "#1B5E20"
COLOR_MAX = "#F57F17"
COLOR_MIN = "#B71C1C"
COLOR_STD = "#0D47A1"

# rutas a modelos
STACK_MODEL_PATH = "data/modelo_stack.pkl"
LSTM_MODEL_PATH = "data/modelo_lstm.h5"

# zoom inicial del mapa
DEFAULT_MAP_ZOOM = 12
DEFAULT_MAP_TILES = "cartodbpositron"

# bounding box (por si falla el fit_bounds)
DEFAULT_LATLON = [37.620, -0.980]
