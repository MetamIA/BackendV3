"""
Flask App Modulare - Demand Forecasting Backend
Integra tutti i moduli: QueryParser, PredictionManager, MLPredictor
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Add parent directory to path
import sys
import os

# Aggiungi la directory parent (root del progetto) al path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from modules.query_parser import QueryParser
from modules.prediction_manager import PredictionManager
from config.config import (
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    OPENAI_API_KEY
)

# Inizializza Flask
app = Flask(__name__)
CORS(app)

# Verifica API key
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY non trovata! "
        "Crea un file .env con: OPENAI_API_KEY=sk-your-key"
    )

# Inizializza i moduli
print("\n" + "="*60)
print("🚀 INIZIALIZZAZIONE SISTEMA DEMAND FORECASTING")
print("="*60)

try:
    query_parser = QueryParser()
    print("✅ Modulo 1: Query Parser inizializzato")
except Exception as e:
    print(f"❌ Errore Query Parser: {e}")
    query_parser = None

try:
    prediction_manager = PredictionManager()
    print("✅ Modulo 2: Prediction Manager inizializzato")
    print("✅ Modulo 3: Trends Analyzer inizializzato")
except Exception as e:
    print(f"❌ Errore Prediction Manager: {e}")
    prediction_manager = None

print("="*60 + "\n")

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Endpoint principale per processare messaggi dell'utente

    Flow:
    1. Query Parser determina se è chat o query
    2. Se query → Prediction Manager gestisce predizioni e risposta
    3. Se chat → Risposta diretta
    """
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({
                "error": "Campo 'message' mancante"
            }), 400

        user_message = data['message']

        if not query_parser:
            return jsonify({
                "error": "Query Parser non disponibile"
            }), 503

        # STEP 1: Parse del messaggio (Modulo 1)
        parse_result = query_parser.parse_message(user_message)

        if parse_result.get('tipo') == 'error':
            return jsonify(parse_result), 500

        # STEP 2a: Se è chat normale, ritorna risposta diretta
        if parse_result.get('tipo') == 'chat':
            return jsonify({
                "tipo": "chat",
                "risposta": parse_result.get('risposta_chat'),
                "timestamp": datetime.now().isoformat()
            }), 200

        # STEP 2b: Se è query, processa con Prediction Manager (Modulo 2)
        if parse_result.get('tipo') == 'query':
            if not prediction_manager:
                return jsonify({
                    "error": "Prediction Manager non disponibile"
                }), 503

            query_combinations = parse_result.get('query_combinations', [])

            if not query_combinations:
                return jsonify({
                    "tipo": "chat",
                    "risposta": "Non ho trovato informazioni sufficienti per generare una predizione. "
                               "Specifica prodotto, cliente e periodo (mese/anno).",
                    "timestamp": datetime.now().isoformat()
                }), 200

            # Processa le query con il Prediction Manager
            prediction_result = prediction_manager.process_query_combinations(
                query_combinations
            )

            return jsonify({
                "tipo": "query",
                "query_info": {
                    "num_combinations": len(query_combinations),
                    "combinations": query_combinations
                },
                "predizioni": prediction_result['risultati'],
                "risposta": prediction_result['risposta'],
                "timestamp": prediction_result['timestamp']
            }), 200

        # Caso non previsto
        return jsonify({
            "error": "Tipo di messaggio non riconosciuto"
        }), 400

    except Exception as e:
        print(f"❌ Errore in /api/chat: {e}")
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict_direct():
    """
    Endpoint diretto per richiedere predizioni senza parsing NLP

    Body:
    {
        "prodotto": "40000",
        "cliente": "393",
        "anno": 2024,
        "mese": 1
    }
    """
    try:
        data = request.get_json()

        required = ['prodotto', 'cliente', 'anno', 'mese']
        if not all(k in data for k in required):
            return jsonify({
                "error": f"Campi richiesti: {', '.join(required)}"
            }), 400

        if not prediction_manager:
            return jsonify({
                "error": "Prediction Manager non disponibile"
            }), 503

        # Crea query combination manuale
        query_combinations = [{
            'prodotto': str(data['prodotto']),
            'cliente': str(data['cliente']),
            'periodo': {
                'mese': int(data['mese']),
                'anno': int(data['anno'])
            }
        }]

        result = prediction_manager.process_query_combinations(query_combinations)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Endpoint per verificare lo stato del sistema"""

    status = {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "query_parser": query_parser is not None,
            "prediction_manager": prediction_manager is not None,
            "ml_predictor": prediction_manager.predictor.is_ready() if prediction_manager else False
        },
        "data": {
            "csv_loaded": prediction_manager.df_predictions is not None if prediction_manager else False,
            "csv_rows": len(prediction_manager.df_predictions) if prediction_manager and prediction_manager.df_predictions is not None else 0
        }
    }

    # Determina se il sistema è in uno stato degradato
    critical_modules = ["query_parser", "prediction_manager"]
    if not all(status["modules"][m] for m in critical_modules):
        status["status"] = "degraded"

    status_code = 200 if status["status"] == "operational" else 503
    return jsonify(status), status_code


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check semplice"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/', methods=['GET'])
def root():
    """Root endpoint con informazioni API"""
    return jsonify({
        "name": "Demand Forecasting API",
        "version": "2.0",
        "endpoints": {
            "POST /api/chat": "Invia un messaggio (chat o query di forecasting)",
            "POST /api/predict": "Richiesta diretta di predizione",
            "GET /api/status": "Stato del sistema",
            "GET /api/health": "Health check"
        }
    }), 200


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 AVVIO SERVER FLASK")
    print("="*60)
    print(f"Host: {FLASK_HOST}")
    print(f"Port: {FLASK_PORT}")
    print(f"Debug: {FLASK_DEBUG}")
    print("="*60 + "\n")

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
