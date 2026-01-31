# 🔀 Logica di Trigger e Decisione Moduli - Dettaglio Completo

## 📊 Overview del Processo Decisionale

```
QUERY UTENTE
    ↓
┌─────────────────────────────────────┐
│  LLM Orchestrator Enhanced          │
│  (Analisi con GPT-4o)               │
└─────────────────┬───────────────────┘
                  ↓
         ┌────────────────┐
         │ STEP 1:        │
         │ Entity         │
         │ Extraction     │
         └────────┬───────┘
                  ↓
         ┌────────────────┐
         │ STEP 2:        │
         │ Risoluzione    │
         │ Nomi→Codici    │
         └────────┬───────┘
                  ↓
         ┌────────────────┐
         │ STEP 3:        │
         │ Decisione      │
         │ Tipo Request   │
         └────────┬───────┘
                  ↓
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌─────────┐              ┌──────────────┐
│  CHAT   │              │ PREDICTION   │
└─────────┘              └──────┬───────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
        ┌──────────────────┐    ┌─────────────────────┐
        │ Statistics Agent │    │ Prediction Manager  │
        │ (Periodo Passato)│    │ (Periodo Futuro)    │
        └──────────────────┘    └─────────────────────┘
```

---

## 🎯 STEP 1: Entity Extraction

### Input
```
Query: "Quanto ho venduto di Osvego 3500 a gennaio 2024?"
```

### LLM Prompt (GPT-4o)
```python
system_prompt = """
Estrai entità dalla query utente:
- Prodotti: Lista nomi prodotti menzionati
- Clienti: Lista nomi clienti menzionati
- Periodo: Periodo temporale (passato/futuro/presente)
"""

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
)
```

### Output
```json
{
    "prodotti": ["Osvego 3500"],
    "clienti": [],
    "periodo_estratto": "gennaio 2024"
}
```

### Casistiche

| Query | Prodotti | Clienti | Periodo |
|-------|----------|---------|---------|
| "Vendite Osvego gennaio 2024" | ["Osvego"] | [] | "gennaio 2024" |
| "Quanto venduto a cliente X?" | [] | ["X"] | null |
| "Predici vendite marzo 2025" | [] | [] | "marzo 2025" |
| "Top 10 prodotti" | [] | [] | null |
| "Ciao!" | [] | [] | null |

---

## 🔍 STEP 2: Risoluzione Entità

### A. Risoluzione Prodotti

```python
# Input
prodotti_input = ["Osvego 3500"]

# Process
for prodotto_nome in prodotti_input:
    # Fuzzy search nel name_code_mapper
    matches = name_code_mapper.search_product(prodotto_nome)
    
    # matches = [
    #   {
    #     "nome": "Osvego 3500 - 14pz x 250g",
    #     "codice": "40000",
    #     "score": 95
    #   },
    #   {
    #     "nome": "Osvego ai 5 Cereali 3500",
    #     "codice": "40006",
    #     "score": 75
    #   }
    # ]
    
    # Prende best match (score >= 70)
    if matches and matches[0]['score'] >= 70:
        codice = matches[0]['codice']  # "40000"
        nome_completo = matches[0]['nome']

# Output
prodotti_codici = ["40000"]
prodotti_nomi = ["Osvego 3500 - 14pz x 250g"]
```

### B. Risoluzione Clienti

```python
# Input
clienti_input = ["DI VITO"]

# Process
for cliente_nome in clienti_input:
    matches = name_code_mapper.search_client(cliente_nome)
    
    # matches = [
    #   {
    #     "nome": "DI VITO RAFFAELE",
    #     "codice": "10001161",
    #     "score": 90
    #   }
    # ]
    
    if matches and matches[0]['score'] >= 70:
        codice = matches[0]['codice']

# Output
clienti_codici = ["10001161"]
clienti_nomi = ["DI VITO RAFFAELE"]
```

### C. Parsing Periodo

```python
# Input
periodo_str = "gennaio 2024"

# Regex patterns
patterns = {
    "mese_anno": r"(gennaio|febbraio|...) (\d{4})",
    "anno_solo": r"(\d{4})",
    "mese_solo": r"(gennaio|febbraio|...)",
}

# Output
periodo = {
    "mese": 1,      # gennaio
    "anno": 2024
}
```

---

## 🚦 STEP 3: Decisione Tipo Richiesta

### Algoritmo Decisionale

```python
def decide_tipo_richiesta(entita, periodo):
    """
    Decide tipo richiesta basandosi su entità estratte
    
    Returns:
        - "prediction": Query con dati strutturati
        - "chat": Conversazione generica
    """
    
    ha_prodotti = len(entita['prodotti']) > 0
    ha_clienti = len(entita['clienti']) > 0
    ha_periodo = periodo is not None
    
    # CASO 1: Query generica senza entità
    if not ha_prodotti and not ha_clienti and not ha_periodo:
        # "Ciao", "Come funziona?", "Aiutami"
        return "chat"
    
    # CASO 2: Ha entità (prodotti/clienti/periodo)
    if ha_prodotti or ha_clienti or ha_periodo:
        # "Vendite Osvego 2024", "Predici marzo 2025"
        return "prediction"
    
    # CASO 3: Default fallback
    return "prediction"
```

### Matrice Decisionale

| Prodotti | Clienti | Periodo | Tipo | Modulo Target |
|----------|---------|---------|------|---------------|
| ✅ | - | ✅ Passato | `prediction` | Statistics Agent |
| ✅ | - | ✅ Futuro | `prediction` | Prediction Manager |
| - | ✅ | ✅ Passato | `prediction` | Statistics Agent |
| - | ✅ | ✅ Futuro | `prediction` | Prediction Manager |
| ✅ | ✅ | ✅ | `prediction` | Statistics/Prediction |
| - | - | - | `chat` | Orchestrator |
| ✅ | - | ❌ | `prediction` | Statistics (all time) |

---

## 🎛️ STEP 4: Decisione Operazione

### Se tipo="prediction"

```python
def decide_operazione(prodotti, clienti, periodo, query):
    """
    Decide quale operazione prediction eseguire
    """
    
    ha_prodotti = len(prodotti) > 0
    ha_clienti = len(clienti) > 0
    
    # Check keywords nella query
    is_top_query = "top" in query.lower() or "migliori" in query.lower()
    is_cluster_query = "cluster" in query.lower()
    
    # DECISIONE
    if is_top_query:
        if "prodotti" in query.lower():
            return "top_prodotti"
        elif "clienti" in query.lower():
            return "top_clienti"
    
    if is_cluster_query:
        return "analisi_cluster"
    
    if ha_prodotti and not ha_clienti:
        return "aggregazione_clienti"  # Somma per tutti i clienti
    
    if ha_clienti and not ha_prodotti:
        return "aggregazione_prodotti"  # Somma per tutti i prodotti
    
    if ha_prodotti and ha_clienti:
        return "predizione_singola"  # Predizione specifica
    
    return "aggregazione_clienti"  # Default
```

### Matrice Operazioni

| Query Pattern | Operazione | Output |
|---------------|------------|--------|
| "Vendite Osvego 2024" | `aggregazione_clienti` | Somma tutti clienti |
| "Vendite cliente X" | `aggregazione_prodotti` | Somma tutti prodotti |
| "Top 10 prodotti" | `top_prodotti` | Classifica prodotti |
| "Top 10 clienti" | `top_clienti` | Classifica clienti |
| "Analisi cluster rank 3" | `analisi_cluster` | Dati per cluster |
| "Predici Osvego a cliente X marzo" | `predizione_singola` | Predizione specifica |

---

## 🔄 STEP 5: Costruzione Piano Esecuzione

### Esempio 1: Query Statistiche

**Input:**
```
Query: "Quanto ho venduto di Osvego 3500 a gennaio 2024?"
Entità estratte: {prodotti: ["Osvego 3500"], periodo: "gennaio 2024"}
Tipo: "prediction"
Operazione: "aggregazione_clienti"
```

**Piano Generato:**
```json
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
                "prodotti": ["40000"],
                "clienti": "*",
                "periodo": {
                    "mese": 1,
                    "anno": 2024
                }
            }
        }
    }
}
```

### Esempio 2: Query Chat

**Input:**
```
Query: "Ciao, come funziona il sistema?"
Entità estratte: {}
Tipo: "chat"
```

**Piano Generato:**
```json
{
    "tipo_richiesta": "chat",
    "risposta_diretta": "Ciao! Il sistema permette di analizzare..."
}
```

### Esempio 3: Predizione Futuro

**Input:**
```
Query: "Predici vendite Osvego marzo 2025"
Entità estratte: {prodotti: ["Osvego"], periodo: "marzo 2025"}
Tipo: "prediction"
```

**Piano Generato:**
```json
{
    "tipo_richiesta": "prediction",
    "moduli": {
        "prediction": {
            "attivo": true,
            "operazione": "aggregazione_clienti",
            "input": {
                "prodotti": ["40000"],
                "clienti": "*",
                "periodo": {
                    "mese": 3,
                    "anno": 2025
                }
            }
        }
    }
}
```

---

## 🎯 STEP 6: Esecuzione Modulo

### A. Se tipo="chat" → Orchestrator Risponde

```python
if piano['tipo_richiesta'] == 'chat':
    risposta = piano['risposta_diretta']
    return risposta
```

### B. Se tipo="prediction" → Routing a Modulo

```python
if piano['tipo_richiesta'] == 'prediction':
    prediction_config = piano['moduli']['prediction']
    
    # Estrai parametri
    operazione = prediction_config['operazione']
    input_data = prediction_config['input']
    
    # Chiama Prediction Manager
    result = prediction_manager.execute(
        operazione=operazione,
        input_data=input_data,
        user_query=query_originale
    )
```

### C. Prediction Manager → Delega

```python
class PredictionManager:
    def execute(self, operazione, input_data, user_query):
        periodo = input_data['periodo']
        anno = periodo['anno']
        
        # DECISIONE: Passato vs Futuro
        if anno <= 2024:
            # PASSATO → Statistics Agent
            return self._delega_statistics_agent(
                user_query=user_query,
                prodotti_codici=input_data['prodotti']
            )
        else:
            # FUTURO → Prediction con XGBoost
            return self._esegui_predizione(
                operazione=operazione,
                input_data=input_data
            )
    
    def _delega_statistics_agent(self, user_query, prodotti_codici):
        """Delega a Statistics Agent per query passato"""
        
        result = self.statistics_agent.query_statistics(
            user_query=user_query,
            conversation_history=self.conversation_history,
            prodotti_codici=prodotti_codici  # ← Codici già risolti!
        )
        
        return result
```

---

## 🔧 Statistics Agent - Flow Interno

### Tool Calling Architecture

```
Statistics Agent
    │
    ├─ System Prompt
    │   - Hai 3 tools
    │   - Puoi chiedere chiarimenti
    │   - Usa RAG per cercare dati
    │
    ├─ Tool 1: search_csv_semantic
    │   Input: query naturale
    │   Process:
    │     1. Estrai filtri da query (anno, mese)
    │     2. Usa prodotti_codici per filtro esatto
    │     3. ChromaDB search con filtri
    │   Output: metadata + summary
    │
    ├─ Tool 2: calculate_aggregate
    │   Input: data + operation + field
    │   Process: Calcola su metadata
    │   Output: valore aggregato
    │
    └─ Tool 3: ask_clarification
        Input: question + suggestions
        Output: domanda all'utente
```

### Iterazioni Tool Calling

```python
# ITERAZIONE 1: LLM decide di cercare
llm_response = {
    "tool_calls": [{
        "name": "search_csv_semantic",
        "arguments": {
            "query": "vendite Osvego 3500 gennaio 2024"
        }
    }]
}

# Esegui tool
tool_result = search_csv_semantic(
    query="vendite Osvego 3500 gennaio 2024"
)

# Filtri applicati automaticamente:
# {
#     "$and": [
#         {"prodotto_codice": {"$eq": "40000"}},  # Da prodotti_codici
#         {"esercizio": {"$eq": 2024}},
#         {"periodo": {"$eq": 1}}
#     ]
# }

# Result:
# {
#     "num_results": 77,
#     "summary": {
#         "kg_totale": 3057.33,
#         "ricavo_totale": 15447.57
#     }
# }

# ITERAZIONE 2: LLM formula risposta
llm_response = {
    "content": "Nel gennaio 2024, hai venduto 3,057.33 kg di Osvego 3500"
}
```

---

## 🎨 Esempi Completi End-to-End

### Esempio 1: Vendite Passato

```
INPUT: "Quanto ho venduto di Osvego 3500 a gennaio 2024?"

STEP 1 - Entity Extraction:
  prodotti: ["Osvego 3500"]
  clienti: []
  periodo: "gennaio 2024"

STEP 2 - Risoluzione:
  prodotti_codici: ["40000"]
  periodo: {mese: 1, anno: 2024}

STEP 3 - Tipo Richiesta:
  tipo: "prediction" (ha prodotti + periodo)

STEP 4 - Operazione:
  operazione: "aggregazione_clienti" (ha prodotti, no clienti)

STEP 5 - Piano:
  {
    "tipo": "prediction",
    "moduli": {
      "prediction": {
        "operazione": "aggregazione_clienti",
        "input": {
          "prodotti": ["40000"],
          "clienti": "*",
          "periodo": {mese: 1, anno: 2024}
        }
      }
    }
  }

STEP 6 - Esecuzione:
  → Prediction Manager
    → anno=2024 → PASSATO
      → Delega Statistics Agent
        → Tool: search_csv_semantic
          → Filtri: {prodotto: "40000", anno: 2024, mese: 1}
            → ChromaDB: 77 righe
              → Summary: kg=3057.33
                → LLM: "Nel gennaio 2024, hai venduto 3,057.33 kg"

OUTPUT: "Nel gennaio 2024, hai venduto 3,057.33 kg di Osvego 3500..."
```

### Esempio 2: Predizione Futuro

```
INPUT: "Predici vendite Osvego marzo 2025"

STEP 1-5: Similar...

STEP 6 - Esecuzione:
  → Prediction Manager
    → anno=2025 → FUTURO
      → Usa XGBoost Predictor
        → Features: prodotto=40000, periodo=3, anno=2025
          → Predizione ML: kg=245.67
            → Statistics Agent formatta
              → LLM: "Per marzo 2025, prevedo vendite di 245.67 kg"

OUTPUT: "Per marzo 2025, prevedo vendite di 245.67 kg di Osvego 3500"
```

### Esempio 3: Chat Generica

```
INPUT: "Ciao, come funziona il sistema?"

STEP 1 - Entity Extraction:
  prodotti: []
  clienti: []
  periodo: null

STEP 3 - Tipo Richiesta:
  tipo: "chat" (nessuna entità)

STEP 6 - Esecuzione:
  → Orchestrator risponde direttamente
    → GPT-4o conversazionale

OUTPUT: "Ciao! Il sistema ti permette di analizzare vendite..."
```

---

## 🎯 Decision Points Riassuntivi

### 1. TIPO RICHIESTA
```
Ha entità? (prodotti/clienti/periodo)
├─ NO → tipo="chat"
└─ SÌ → tipo="prediction"
```

### 2. OPERAZIONE (se prediction)
```
Query contiene...
├─ "top prodotti" → top_prodotti
├─ "top clienti" → top_clienti
├─ "cluster" → analisi_cluster
├─ Solo prodotti → aggregazione_clienti
├─ Solo clienti → aggregazione_prodotti
└─ Prodotti + clienti → predizione_singola
```

### 3. MODULO TARGET (se prediction)
```
Periodo...
├─ <= 2024 → Statistics Agent (RAG su CSV)
└─ >= 2025 → Prediction Manager (XGBoost ML)
```

### 4. TOOL STATISTICS AGENT
```
Confidenza...
├─ Alta → search_csv_semantic + calculate_aggregate
├─ Media → search_csv_semantic
└─ Bassa → ask_clarification
```

---

**Documentazione completa della logica di trigger e decisione! 🎯**
