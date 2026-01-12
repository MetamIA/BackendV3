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

# Google Trends
USE_GOOGLE_TRENDS = True  # Abilita analisi Google Trends
TRENDS_TIMEOUT = 10  # Timeout in secondi per richieste trends

# RAG System (Modulo 4)
USE_RAG_SYSTEM = True  # Abilita/disabilita RAG System
RAG_DOCUMENTS_DIR = str(DATA_DIR / "documents")  # Cartella documenti aziendali
RAG_N_RESULTS = 3  # Numero di risultati da recuperare per query
RAG_CHUNK_SIZE = 500  # Dimensione chunk per testo
RAG_CHUNK_OVERLAP = 50  # Overlap tra chunk

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
