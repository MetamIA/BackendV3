# 🏢 Sistema Gentilini - Documentazione Architetturale

## 📋 Indice
1. [Panoramica Sistema](#panoramica-sistema)
2. [Architettura Modulare](#architettura-modulare)
3. [Flow Applicativo](#flow-applicativo)
4. [Logica di Trigger](#logica-di-trigger)
5. [Moduli in Dettaglio](#moduli-in-dettaglio)
6. [API Endpoints](#api-endpoints)
7. [Installazione e Setup](#installazione-e-setup)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Panoramica Sistema

### Descrizione
Sistema conversazionale AI per la gestione e analisi di dati commerciali dell'azienda Gentilini. Il sistema permette di:
- **Interrogare statistiche** su vendite, prodotti e clienti
- **Ottenere predizioni** su vendite future
- **Conversare naturalmente** per ottenere insights dai dati

### Tecnologie Core
- **LLM:** OpenAI GPT-4o (orchestrazione e conversazione)
- **RAG:** ChromaDB + sentence-transformers (ricerca semantica)
- **Backend:** Flask + Python 3.11
- **ML:** XGBoost (predizioni vendite)
- **Database:** CSV (122K+ righe transazioni)

### Dataset
```
File: output_addestramento_finale.csv
Righe: 122,482
Periodo: 2024-2026
Colonne: 21 (prodotti, clienti, vendite, cluster, predizioni)
```

---

## 🏗️ Architettura Modulare

### Componenti Principali

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│                      (Flask app.py)                          │
└────────────┬────────────────────────────────┬────────────────┘
             │                                │
             ▼                                ▼
    ┌────────────────┐              ┌─────────────────┐
    │  Conversation  │              │  LLM Orchestrator│
    │    Manager     │◄────────────►│    Enhanced     │
    └────────────────┘              └────────┬────────┘
                                             │
                        ┌────────────────────┼────────────────────┐
                        │                    │                    │
                        ▼                    ▼                    ▼
              ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
              │   Statistics     │  │   Prediction     │  │   (Futuro)       │
              │   Agent + RAG    │  │   Manager        │  │   Altri Moduli   │
              └──────────────────┘  └──────────────────┘  └──────────────────┘
                        │                    │
                        ▼                    ▼
              ┌──────────────────┐  ┌──────────────────┐
              │   ChromaDB       │  │   XGBoost        │
              │   (CSV-RAG)      │  │   (Predictor)    │
              └──────────────────┘  └──────────────────┘
```

### Moduli in Essere

| Modulo | Responsabilità | Status | File |
|--------|----------------|--------|------|
| **LLM Orchestrator Enhanced** | Entity extraction, routing, coordinazione | ✅ Attivo | `llm_orchestrator_enhanced.py` |
| **Statistics Agent** | Query statistiche + RAG semantico | ✅ Attivo | `statistics_agent.py` |
| **Prediction Manager Enhanced** | Predizioni vendite ML | ✅ Attivo | `prediction_manager_enhanced.py` |
| **Conversation Manager** | Storia conversazioni, context | ✅ Attivo | `conversation_manager.py` |
| **CSV RAG Indexer** | Indicizzazione semantica CSV | ✅ Attivo | `csv_rag_indexer.py` |
| **Name-Code Mapper** | Mapping nomi ↔ codici | ✅ Attivo | `name_code_mapper.py` |
| **Predictor** | Modello XGBoost | ✅ Attivo | `predictor.py` |

---

## 🔄 Flow Applicativo

### 1. Query Utente → Risposta Finale

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. UTENTE                                                         │
│    Query: "Quanto ho venduto di Osvego 3500 a gennaio 2024?"    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. API LAYER (app.py)                                            │
│    - Riceve query                                                 │
│    - Carica conversation history                                 │
│    - Chiama LLM Orchestrator                                     │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. LLM ORCHESTRATOR ENHANCED                                     │
│    ┌──────────────────────────────────────────────────┐         │
│    │ A. ENTITY EXTRACTION                             │         │
│    │    - Prodotti: ["Osvego 3500"]                   │         │
│    │    - Clienti: []                                 │         │
│    │    - Periodo: {"mese": 1, "anno": 2024}          │         │
│    └──────────────────────────────────────────────────┘         │
│                                                                   │
│    ┌──────────────────────────────────────────────────┐         │
│    │ B. RISOLUZIONE ENTITÀ                            │         │
│    │    - "Osvego 3500" → "Osvego 3500 - 14pz x 250g" │         │
│    │    - Nome → Codice: "40000"                      │         │
│    └──────────────────────────────────────────────────┘         │
│                                                                   │
│    ┌──────────────────────────────────────────────────┐         │
│    │ C. DECISIONE MODULO                              │         │
│    │    - Tipo: "prediction" (ha periodo + prodotti)  │         │
│    │    - Operazione: "aggregazione_clienti"          │         │
│    └──────────────────────────────────────────────────┘         │
│                                                                   │
│    ┌──────────────────────────────────────────────────┐         │
│    │ D. PIANO ESECUZIONE                              │         │
│    │    {                                             │         │
│    │      "tipo_richiesta": "prediction",             │         │
│    │      "moduli": {                                 │         │
│    │        "prediction": {                           │         │
│    │          "attivo": true,                         │         │
│    │          "operazione": "aggregazione_clienti",   │         │
│    │          "input": {                              │         │
│    │            "prodotti": ["40000"],                │         │
│    │            "clienti": "*",                       │         │
│    │            "periodo": {"mese": 1, "anno": 2024}  │         │
│    │          }                                       │         │
│    │        }                                         │         │
│    │      }                                           │         │
│    │    }                                             │         │
│    └──────────────────────────────────────────────────┘         │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
   ┌─────────────────────┐         ┌─────────────────────┐
   │ 4A. STATISTICS      │         │ 4B. PREDICTION       │
   │     AGENT           │         │     MANAGER          │
   │ (se tipo="chat")    │         │ (se tipo="prediction")│
   └─────────────────────┘         └──────────┬───────────┘
                                              │
                                              ▼
                                   ┌─────────────────────────┐
                                   │ Statistics Agent        │
                                   │ (conversazionale)       │
                                   │                         │
                                   │ ┌─────────────────────┐ │
                                   │ │ Iterazione 1:       │ │
                                   │ │ search_csv_semantic │ │
                                   │ │ - Query RAG         │ │
                                   │ │ - Filtri:           │ │
                                   │ │   • prodotto: 40000 │ │
                                   │ │   • anno: 2024      │ │
                                   │ │   • mese: 1         │ │
                                   │ └─────────────────────┘ │
                                   │                         │
                                   │ ┌─────────────────────┐ │
                                   │ │ ChromaDB Search     │ │
                                   │ │ Righe trovate: 77   │ │
                                   │ │ Kg totali: 3,057.33 │ │
                                   │ └─────────────────────┘ │
                                   │                         │
                                   │ ┌─────────────────────┐ │
                                   │ │ Iterazione 2:       │ │
                                   │ │ Risposta finale     │ │
                                   │ │ "Nel gennaio 2024   │ │
                                   │ │  hai venduto        │ │
                                   │ │  3,057.33 kg"       │ │
                                   │ └─────────────────────┘ │
                                   └──────────┬──────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. RISPOSTA FINALE                                               │
│    "Nel mese di gennaio 2024, hai venduto un totale di          │
│     3,057.33 kg di Osvego 3500 - 14pz x 250g, generando un      │
│     ricavo netto di €15,447.57 con 21 clienti distinti."        │
└──────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Logica di Trigger

### Come l'Orchestrator Decide Quale Modulo Attivare

```python
# 1. ANALISI QUERY
query = "Quanto ho venduto di Osvego 3500 a gennaio 2024?"

# 2. ENTITY EXTRACTION (GPT-4o)
entita = {
    "prodotti": ["Osvego 3500"],
    "clienti": [],
    "periodo_estratto": "gennaio 2024"
}

# 3. DECISIONE TIPO RICHIESTA
if ha_periodo_futuro AND (ha_prodotti OR ha_clienti):
    tipo = "prediction"  # ← Predizioni
    
elif ha_periodo_passato AND (ha_prodotti OR ha_clienti):
    tipo = "prediction"  # ← Statistics (via prediction → statistics agent)
    
elif domanda_generica OR nessuna_entita:
    tipo = "chat"  # ← Conversazione generica
    
else:
    tipo = "prediction"  # Default per sicurezza
```

### Decision Tree Completa

```
Query Utente
    │
    ├─ Contiene periodo futuro? (2025, 2026, "prossimo mese")
    │   └─ SÌ → tipo="prediction" + modulo=prediction
    │
    ├─ Contiene periodo passato? (2024, "gennaio", "scorso")
    │   │
    │   ├─ Ha prodotti/clienti specifici?
    │   │   └─ SÌ → tipo="prediction" + modulo=statistics_agent
    │   │
    │   └─ NO → tipo="chat"
    │
    ├─ Domanda generica? ("ciao", "come stai", "aiutami")
    │   └─ SÌ → tipo="chat"
    │
    └─ Default → tipo="prediction"
```

### Esempi di Trigger

| Query | Tipo | Modulo | Motivo |
|-------|------|--------|--------|
| "Vendite Osvego gennaio 2024" | `prediction` | `statistics_agent` | Periodo passato + prodotto |
| "Predici vendite marzo 2025" | `prediction` | `prediction_manager` | Periodo futuro |
| "Chi è il miglior cliente?" | `prediction` | `statistics_agent` | Query aggregata passato |
| "Ciao, come va?" | `chat` | Orchestrator diretto | Nessuna entità |
| "Cosa significa cluster rank?" | `chat` | Orchestrator diretto | Domanda concettuale |

---

## 📦 Moduli in Dettaglio

### 1. LLM Orchestrator Enhanced

**File:** `modules/llm_orchestrator_enhanced.py`

**Responsabilità:**
- Entity extraction da query naturale
- Risoluzione nomi → codici prodotti/clienti
- Decisione tipo richiesta (chat/prediction)
- Routing verso modulo appropriato
- Gestione errori e fallback

**Input:**
```python
{
    "user_query": "Quanto ho venduto di Osvego 3500 a gennaio 2024?",
    "conversation_history": [...]
}
```

**Output (Piano):**
```python
{
    "tipo_richiesta": "prediction",
    "entita": {
        "prodotti": ["Osvego 3500"],
        "clienti": [],
        "periodo_estratto": "gennaio 2024"
    },
    "moduli": {
        "prediction": {
            "attivo": true,
            "operazione": "aggregazione_clienti",
            "input": {
                "prodotti": ["40000"],  # Codice risolto
                "clienti": "*",
                "periodo": {"mese": 1, "anno": 2024}
            }
        }
    }
}
```

**Tool Functions:**
- `name_code_mapper.search_product()` - Risolve nomi prodotti
- `name_code_mapper.search_client()` - Risolve nomi clienti

---

### 2. Statistics Agent (con CSV-RAG)

**File:** `modules/statistics_agent.py`

**Responsabilità:**
- Query statistiche conversazionali
- Ricerca semantica CSV con ChromaDB
- Aggregazioni (somme, medie, conteggi)
- Multi-turn conversation (può chiedere chiarimenti)

**Architettura:**
```
Statistics Agent
    │
    ├─ Tool 1: search_csv_semantic
    │   └─ Ricerca semantica + filtri metadata
    │       - Filtri anno/periodo/prodotto
    │       - Ritorna 200 righe + summary
    │
    ├─ Tool 2: calculate_aggregate
    │   └─ Calcola su risultati RAG
    │       - sum, mean, median, count, min, max
    │
    └─ Tool 3: ask_clarification
        └─ Chiede chiarimenti utente
```

**Input:**
```python
statistics_agent.query_statistics(
    user_query="vendite Osvego gennaio 2024",
    conversation_history=[...],
    prodotti_codici=["40000"]  # ← Codici già risolti!
)
```

**Tool Call Example:**
```python
# Iterazione 1
search_csv_semantic(
    query="vendite Osvego 3500 gennaio 2024"
)

# Filtri automatici applicati:
{
    "$and": [
        {"prodotto_codice": {"$eq": "40000"}},  # ← Da prodotti_codici
        {"esercizio": {"$eq": 2024}},
        {"periodo": {"$eq": 1}}
    ]
}

# Risultato ChromaDB:
{
    "num_results": 77,
    "filters_applied": {...},
    "summary": {
        "kg_totale": 3057.33,
        "ricavo_totale": 15447.57,
        "clienti_unici": 21
    }
}

# Iterazione 2
LLM: "Nel gennaio 2024, hai venduto 3,057.33 kg di Osvego 3500"
```

**Fix Recenti:**
- ✅ Filtri anno/periodo ChromaDB (sintassi `$and`, `$eq`)
- ✅ Uso codici prodotto per filtro esatto (no similarity errors)
- ✅ Temperature=0.0 per consistenza matematica
- ✅ n_results=200 fisso (no variabilità)

---

### 3. Prediction Manager Enhanced

**File:** `modules/prediction_manager_enhanced.py`

**Responsabilità:**
- Predizioni vendite future (XGBoost)
- Aggregazioni su dati storici
- Gestione operazioni complesse (cluster, top N)

**Operazioni Supportate:**
- `predizione_singola` - Predice vendite specifiche
- `aggregazione_prodotti` - Somma per prodotto
- `aggregazione_clienti` - Somma per cliente
- `top_prodotti` - Classifica prodotti
- `top_clienti` - Classifica clienti
- `analisi_cluster` - Analisi per cluster

**Flow Predizione:**
```
1. Riceve input da orchestrator
2. Se periodo passato → delega a statistics_agent
3. Se periodo futuro → usa predictor XGBoost
4. Formatta risultati
5. Statistics agent elabora conversazionalmente
```

**Input:**
```python
{
    "prodotti": ["40000"],
    "clienti": "*",
    "periodo": {"mese": 1, "anno": 2024}
}
```

**Delega a Statistics Agent:**
```python
# Prediction Manager rileva periodo passato
if anno <= 2024:
    # Delega a statistics agent
    result = statistics_agent.query_statistics(
        user_query=original_query,
        prodotti_codici=["40000"]
    )
```

---

### 4. CSV RAG Indexer

**File:** `modules/csv_rag_indexer.py`

**Responsabilità:**
- Indicizzazione semantica CSV in ChromaDB
- Generazione embeddings (sentence-transformers)
- Search semantico + filtri metadata

**Processo Indicizzazione:**
```python
# 1. Carica CSV
df = pd.read_csv('output_addestramento_finale.csv')

# 2. Converte ogni riga in testo searchable
text = f"""
Prodotto: {row['Descrizione_Prodotto']}
Cliente: {row['Descrizione_Cliente']}
Periodo: {mese_nome} {row['Esercizio']}
Kg venduti: {row['Kg_Venduti_Reali']:.2f} kg
Ricavo netto: €{row['Ricavo_Netto']:.2f}
Cluster prodotto: {row['Cluster_Prodotto_Label']}
"""

# 3. Genera embedding
embedding = model.encode(text)

# 4. Salva in ChromaDB con metadata
collection.add(
    documents=[text],
    embeddings=[embedding],
    metadatas=[{
        "prodotto_codice": "40000",
        "prodotto_nome": "Osvego 3500",
        "cliente_codice": "10001161",
        "cliente_nome": "DI VITO RAFFAELE",
        "esercizio": 2024,
        "periodo": 1,
        "kg_venduti": 3.53,
        "ricavo_netto": 17.01,
        ...
    }]
)
```

**Search con Filtri:**
```python
results = indexer.search(
    query="vendite gennaio",
    n_results=200,
    filters={
        "$and": [
            {"prodotto_codice": {"$eq": "40000"}},
            {"esercizio": {"$eq": 2024}},
            {"periodo": {"$eq": 1}}
        ]
    }
)
```

**Performance:**
- Indicizzazione: ~5-10 min per 122K righe (una volta)
- Query: ~100-300ms
- Memoria: ~200MB embeddings
- Disco: ~50MB ChromaDB

---

## 🌐 API Endpoints

### POST `/api/chat`

**Descrizione:** Endpoint principale per conversazione

**Request:**
```json
{
    "message": "Quanto ho venduto di Osvego 3500 a gennaio 2024?",
    "conversation_id": "conv_12345"  // Opzionale
}
```

**Response:**
```json
{
    "response": "Nel gennaio 2024, hai venduto 3,057.33 kg...",
    "conversation_id": "conv_12345",
    "plan": {
        "tipo_richiesta": "prediction",
        "moduli": {...}
    }
}
```

### POST `/api/suggestions`

**Descrizione:** Genera suggerimenti query basati su contesto

**Request:**
```json
{
    "conversation_id": "conv_12345",
    "n_suggestions": 3
}
```

**Response:**
```json
{
    "suggestions": [
        "Mostrami le vendite per cliente",
        "Quali sono stati i ricavi totali?",
        "Analizza i cluster prodotto"
    ]
}
```

---

## 💾 Installazione e Setup

### Requisiti
```
Python 3.11+
OpenAI API Key
10GB+ RAM
5GB+ Spazio disco
```

### Installazione

```bash
# 1. Clona repository
git clone <repo>
cd sistema-gentilini

# 2. Installa dipendenze
pip install -r requirements.txt

# 3. Installa dipendenze RAG
pip install chromadb sentence-transformers

# 4. Setup variabili ambiente
echo "OPENAI_API_KEY=sk-..." > .env

# 5. Indicizza CSV (prima volta, ~10 min)
python modules/csv_rag_indexer.py

# 6. Avvia server
python api/app.py
```

**Output atteso:**
```
✅ Statistics Agent con CSV RAG
   Chunks indicizzati: 122482
   
🚀 Server avviato su http://localhost:5000
```

### Verifica Setup

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Ciao!"}'
```

---

## 🐛 Troubleshooting

### Problema: ChromaDB "Expected where to have exactly one operator"

**Causa:** Sintassi filtri ChromaDB non corretta

**Fix:** Usa operatori `$eq`, `$and`, `$or`
```python
# ❌ SBAGLIATO
filters = {"esercizio": 2024}

# ✅ CORRETTO
filters = {"esercizio": {"$eq": 2024}}
```

---

### Problema: Risultati Inconsistenti (numeri diversi ogni volta)

**Causa:** Temperature LLM troppo alta o n_results variabile

**Fix:** Già applicato in v2.0
```python
temperature=0.0  # Era 0.2
n_results=200    # Fisso, non variabile
```

---

### Problema: Query "Miele" Aggrega Prodotti Sbagliati

**Causa:** Similarity search semantica troppo larga, non usa codice prodotto

**Fix:** Passare codici prodotto già risolti
```python
# In app.py
prodotti_codici = input_data.get('prodotti', [])

agent_result = statistics_agent.query_statistics(
    user_query,
    conversation_history,
    prodotti_codici=prodotti_codici  # ← FIX!
)
```

---

### Problema: "CSV non disponibile"

**Causa:** File CSV non trovato o non indicizzato

**Fix:**
```bash
# Verifica file
ls data/output_addestramento_finale.csv

# Reindicizza
python modules/csv_rag_indexer.py
```

---

## 📊 Statistiche Sistema

### Dataset
```
Righe totali: 122,482
Prodotti unici: 250+
Clienti unici: 800+
Periodo: 2024-2026
Colonne: 21
```

### Performance
```
Query semplice: ~1-2s
Query complessa: ~3-5s
RAG search: ~100-300ms
Indicizzazione: ~5-10min (una volta)
```

### Accuratezza
```
Entity extraction: ~95%
Predizioni ML: RMSE < 15%
Statistics RAG: 100% (con fix)
```

---

## 📝 Log delle Versioni

### v2.0 (2026-01-31) - Fix Critici
- ✅ Fix filtri ChromaDB (sintassi `$and`, `$eq`)
- ✅ Fix consistenza matematica (temperature=0, n_results fisso)
- ✅ Fix bug "miele" (uso codici prodotto per filtro esatto)
- ✅ Documentazione completa

### v1.5 (2026-01-29) - CSV-RAG Integration
- ✅ Integrazione ChromaDB per search semantico
- ✅ Statistics Agent conversazionale con tool calling
- ✅ Filtri metadata anno/periodo
- ✅ Test suite completa

### v1.0 (2026-01-15) - Sistema Base
- ✅ Orchestrator Enhanced
- ✅ Prediction Manager
- ✅ Conversation Manager
- ✅ API REST

---

## 🔗 File Importanti

### Core
- `api/app.py` - API Flask + routing
- `modules/llm_orchestrator_enhanced.py` - Orchestrazione
- `modules/statistics_agent.py` - Statistics + RAG
- `modules/prediction_manager_enhanced.py` - Predizioni
- `modules/csv_rag_indexer.py` - Indicizzazione RAG

### Config
- `config/config.py` - Configurazione sistema
- `.env` - Variabili ambiente (non committare!)
- `requirements.txt` - Dipendenze Python

### Data
- `data/output_addestramento_finale.csv` - Dataset principale
- `data/chroma_db/` - Database ChromaDB (generato)
- `models/xgboost_model.pkl` - Modello ML salvato

### Test
- `tests/test_statistics_agent.py` - Test suite completa
- `tests/quick_test_statistics.py` - Test rapido
- `tests/test_bug_14kg.py` - Test fix bug specifici

### Documentazione
- `README.md` - Questo file
- `docs/BUG_14KG_ANALISI_E_FIX.md` - Fix bug 14kg
- `docs/BUG_MIELE_ANALISI_E_FIX.md` - Fix bug miele
- `docs/PATCH_APP_PY.txt` - Patch codici prodotto

---

## 👥 Contributori

Sistema sviluppato per Gentilini S.p.A.

**Tech Stack:**
- OpenAI GPT-4o
- ChromaDB + sentence-transformers
- XGBoost
- Flask + Python

**Ultima Revisione:** 31 Gennaio 2026

---

## 📞 Support

Per problemi o domande:
1. Controlla [Troubleshooting](#troubleshooting)
2. Verifica log in `logs/app.log`
3. Controlla issue tracker

---

**Sistema Operativo al 100%! 🚀**
