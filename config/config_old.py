"""
Configurazione centralizzata per il backend di demand forecasting
"""

import os
from pathlib import Path

# Percorsi base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# File paths
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
MODEL_PATH = MODELS_DIR / "vendite_model.pkl"

# Crea le directory se non esistono
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_QUERY = "gpt-4o-mini"  # Per parsing query (economico)
OPENAI_MODEL_RESPONSE = "gpt-4o-mini"  # Per generare risposte (può essere gpt-4o per qualità)

# Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

# Predizioni
PREDICTION_CONFIDENCE_THRESHOLD = 0.7  # Soglia di confidenza minima
CACHE_PREDICTIONS = True  # Cache predizioni in memoria

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
