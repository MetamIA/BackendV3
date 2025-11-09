# 🔮 AI-Powered Demand Forecasting System

> Sistema intelligente di previsione della domanda con chatbot conversazionale, Machine Learning e analisi Google Trends per Gentilini.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

---

## 📋 Indice

- [Panoramica](#-panoramica)
- [Features](#-features)
- [Architettura](#-architettura)
- [Tech Stack](#-tech-stack)
- [Installazione](#-installazione)
- [Configurazione](#-configurazione)
- [Utilizzo](#-utilizzo)
- [API Endpoints](#-api-endpoints)
- [Struttura Progetto](#-struttura-progetto)
- [Esempi](#-esempi)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)

---

## 🎯 Panoramica

Sistema end-to-end per la previsione della domanda che combina:
- **NLP** per query in linguaggio naturale
- **Machine Learning** per predizioni accurate
- **Google Trends** per analisi fattori esogeni
- **GPT-4** per risposte intelligenti e contestualizzate

### Il Problema

Le aziende di biscotti come Gentilini hanno bisogno di prevedere accuratamente la domanda per:
- Ottimizzare la produzione
- Gestire l'inventario
- Pianificare le campagne marketing
- Analizzare trend di mercato

### La Soluzione

Un chatbot conversazionale che:
1. Comprende query in linguaggio naturale ("Quanto venderò del prodotto X a gennaio?")
2. Genera predizioni basate su dati storici o ML
3. Analizza Google Trends per contesto di mercato
4. Fornisce insights strategici integrati

---

## ✨ Features

### 🤖 Chatbot Conversazionale
- Query in linguaggio naturale (italiano)
- Distinzione automatica chat/query predittiva
- Risposte contestualizzate con GPT-4

### 📊 Predizioni Intelligenti
- 122K+ predizioni precaricate da CSV
- Generazione runtime con ML (GradientBoosting)
- Metriche: R² ~0.85-0.90, MAE ottimizzato
- Features temporali e prodotto/cliente

### 📈 Analisi Google Trends
- Keywords automatiche generate da GPT
- Dati interesse, trend, variazioni
- Insights su stagionalità e mercato
- Integrazione seamless nelle risposte

### 🎨 Frontend Moderno
- UI responsive e intuitiva
- Visualizzazione predizioni + trends
- Real-time feedback
- Mobile-friendly

---

## 🏗️ Architettura

Il sistema è strutturato in **3 moduli indipendenti**:

```
User Query → [Modulo 1: Query Parser] 
                    ↓
            [Modulo 2: Prediction Manager]
                    ↓
            [Modulo 3: Trends Analyzer]
                    ↓
            [GPT-4 Response Generator]
                    ↓
            Risposta Integrata
```

### Moduli Dettagliati

**Modulo 1: Query Parser**
- Parsing linguaggio naturale con GPT
- Estrazione prodotti, clienti, periodo
- Distinzione query/chat
- Generazione combinazioni

**Modulo 2: Prediction Manager**
- Ricerca in CSV (122K righe)
- Generazione ML se necessario
- Gestione cache e fallback
- Coordinazione trends

**Modulo 3: Trends Analyzer**
- Lettura descrizioni prodotti
- GPT genera 5 keywords specifiche
- Query Google Trends API
- Analisi interesse e trend

---

## 🛠️ Tech Stack

### Backend
- **Flask 3.0.0** - Web framework
- **Python 3.11+** - Linguaggio core
- **Flask-CORS** - Gestione CORS

### AI & ML
- **OpenAI GPT-4** - NLP e generazione risposte
- **scikit-learn 1.3.2** - Machine Learning
- **GradientBoostingRegressor** - Modello predittivo
- **pandas 2.1.4** - Data processing
- **numpy 1.26.4** - Calcolo numerico

### APIs & Data
- **pytrends 4.9.2** - Google Trends API
- **CSV** - 122K+ predizioni storiche
- **joblib** - Model persistence

---

## 🚀 Installazione

### Prerequisiti

- Python 3.11 o superiore
- pip (package manager)
- Chiave API OpenAI ([ottienila qui](https://platform.openai.com/api-keys))

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/demand-forecasting-backend.git
cd demand-forecasting-backend

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Configura .env
echo "OPENAI_API_KEY=sk-proj-your-key" > .env

# 5. Avvia server
python api/app.py
```

**Output atteso:**
```
🚀 INIZIALIZZAZIONE SISTEMA DEMAND FORECASTING
✅ Modulo 1: Query Parser inizializzato
✅ Modello GradientBoosting caricato (R²: 0.87)
✅ CSV predizioni caricato: 122475 righe
✅ Modulo 2: Prediction Manager inizializzato
✅ Modulo 3: Trends Analyzer inizializzato

Server running on http://localhost:5000
```

---

## ⚙️ Configurazione

### File .env (Root)

```env
# Obbligatoria
OPENAI_API_KEY=sk-proj-xxxxx

# Opzionali
FLASK_PORT=5000
FLASK_DEBUG=True
```

### File config/config.py

```python
# OpenAI
OPENAI_MODEL_PARSER = "gpt-4o-mini"
OPENAI_MODEL_RESPONSE = "gpt-4o-mini"

# Flask
FLASK_PORT = 5000
FLASK_DEBUG = True

# ML
PREDICTION_CONFIDENCE_THRESHOLD = 0.7
CACHE_PREDICTIONS = True

# Google Trends
USE_GOOGLE_TRENDS = True
TRENDS_TIMEOUT = 10
```

---

## 📖 Utilizzo

### Via Frontend

1. Apri `frontend_updated.html` nel browser
2. Scrivi: *"Previsioni prodotto 40000 cliente 393 gennaio 2024"*
3. Ricevi risposta con predizioni + trends

### Via API (curl)

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Previsioni prodotto 40000 gennaio 2024"}'
```

### Via Python

```python
import requests

response = requests.post(
    "http://localhost:5000/api/chat",
    json={"message": "Previsioni prodotto 40000 gennaio 2024"}
)

print(response.json()['risposta'])
```

---

## 🔌 API Endpoints

### POST /api/chat

**Request:**
```json
{
  "message": "Previsioni prodotto 40000 cliente 393 gennaio 2024"
}
```

**Response:**
```json
{
  "tipo": "query",
  "risposta": "Per il prodotto 40000 prevediamo 13.49 kg...",
  "predizioni": [{
    "prodotto": "40000",
    "cliente": "393",
    "periodo": {"mese": 1, "anno": 2024},
    "predizione": {"kg_predetti": 13.49, "confidenza": 0.84},
    "fonte": "csv"
  }],
  "trends_data": {...}
}
```

### GET /health

Health check endpoint.

---

## 📁 Struttura Progetto

```
demand-forecasting-backend/
├── README.md
├── requirements.txt
├── .env
├── api/
│   └── app.py
├── modules/
│   ├── query_parser.py
│   ├── prediction_manager.py
│   ├── predictor.py
│   └── trends_analyzer.py
├── config/
│   └── config.py
├── models/
│   └── vendite_model.pkl
├── data/
│   └── predictions.csv
└── frontend_updated.html
```

---

## 💡 Esempi

### Query Semplice

**Input:** *"Quanto venderò del prodotto 40000 a gennaio 2024?"*

**Output:**
```
📊 Per prodotto 40000 prevediamo 13.49 kg a gennaio 2024.

🔍 Analisi Mercato:
L'interesse per "biscotti cioccolato" è in crescita (+12%), 
suggerendo una domanda in aumento.

💡 La crescita dell'interesse supporta una performance 
sopra la media nel periodo.
```

### Query Multipla

**Input:** *"Predizioni prodotto 40000 e 40140, cliente 393, marzo 2024"*

**Output:** Predizioni per entrambi i prodotti + trends separati

### Chat Normale

**Input:** *"Ciao, come funzioni?"*

**Output:** Spiegazione funzionalità e guida utilizzo

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'modules'"

```bash
python -m api.app  # Esegui come modulo
```

### "numpy.dtype size changed"

```bash
pip uninstall -y numpy pandas scikit-learn
pip install numpy==1.26.4 pandas==2.1.4 scikit-learn==1.3.2
```

### "OPENAI_API_KEY not found"

Verifica che `.env` esista nella root con `OPENAI_API_KEY=sk-...`

### "Google Trends error 400"

Sistema usa automaticamente dati recenti per periodi futuri

---

## 🗺️ Roadmap

### ✅ v1.0 (Completato)
- [x] Architettura a 3 moduli
- [x] Query parsing NLP
- [x] Predizioni CSV + ML
- [x] Google Trends integration
- [x] Frontend responsive

### 🚧 v1.1 (In Sviluppo)
- [ ] Autenticazione utenti
- [ ] Dashboard analytics
- [ ] Export Excel/PDF
- [ ] Testing automatizzato
- [ ] Docker containerization

### 🔮 v2.0 (Futuro)
- [ ] Analisi competitor
- [ ] Alert automatici
- [ ] Multi-language support
- [ ] Mobile app
- [ ] Real-time predictions

---

## 📄 Licenza

Questo progetto è rilasciato sotto licenza **MIT**.

---

## 👥 Autori

- **Valerio** - [@yourusername](https://github.com/yourusername)

---

## 🙏 Ringraziamenti

- **Gentilini** - Caso d'uso e dati
- **OpenAI** - API GPT-4
- **scikit-learn** - Algoritmi ML
- **Google Trends** - Dati mercato

---

<div align="center">

**[⬆ Torna all'inizio](#-ai-powered-demand-forecasting-system)**

Made with ❤️ and 🤖 by Valerio

</div>
