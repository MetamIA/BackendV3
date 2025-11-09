"""
Flask app per il backend del demand forecasting
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from query_parser import QueryParser

app = Flask(__name__)
CORS(app)  # Abilita CORS per permettere richieste dal frontend

# Inizializza il parser (la chiave API verrà presa dalle variabili d'ambiente)
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY non trovata nelle variabili d'ambiente")

parser = QueryParser(API_KEY)


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Endpoint per processare i messaggi dell'utente
    
    Expected JSON:
    {
        "message": "Il messaggio dell'utente"
    }
    
    Returns JSON con:
    - tipo: "chat" o "query"
    - dati: informazioni estratte (per query)
    - query_combinations: array di combinazioni prodotto x cliente (per query)
    - risposta_chat: risposta testuale (per chat normale)
    """
    try:
        # Ottieni il messaggio dal body della richiesta
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Campo 'message' mancante nel body della richiesta"
            }), 400
        
        user_message = data['message']
        
        # Processa il messaggio
        result = parser.parse_message(user_message)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "tipo": "error"
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint per verificare che il server sia attivo"""
    return jsonify({
        "status": "ok",
        "message": "Il server è attivo e funzionante"
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
