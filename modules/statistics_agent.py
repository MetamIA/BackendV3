"""
Statistics Agent - Conversational CSV analysis with RAG
L'LLM usa RAG per interrogare semanticamente il CSV
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Any
from openai import OpenAI
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import OPENAI_API_KEY, OPENAI_MODEL_QUERY, PREDICTIONS_CSV
from modules.csv_rag_indexer import CSVRAGIndexer


class StatisticsAgent:
    """
    Agent conversazionale per statistiche CSV con RAG
    L'LLM usa retrieval semantico per interrogare il CSV
    """
    
    def __init__(
        self,
        csv_path: str = str(PREDICTIONS_CSV),
        api_key: str = OPENAI_API_KEY,
        use_rag: bool = True
    ):
        """Inizializza agent"""
        self.csv_path = csv_path
        self.client = OpenAI(api_key=api_key)
        self.model = OPENAI_MODEL_QUERY  # gpt-4o-mini
        self.use_rag = use_rag
        
        # Carica CSV (per fallback e statistiche)
        self.df = None
        self._load_csv()
        
        # Inizializza CSV RAG
        if use_rag:
            try:
                self.rag_indexer = CSVRAGIndexer(csv_path=csv_path)
                
                if self.rag_indexer.available:
                    # Indicizza se necessario
                    self.rag_indexer.index_csv()
                    print(f"✅ Statistics Agent con CSV RAG")
                    print(f"   Chunks indicizzati: {self.rag_indexer.collection.count()}")
                else:
                    print("⚠️  CSV RAG non disponibile, uso query pandas")
                    self.use_rag = False
                    self.rag_indexer = None
            except Exception as e:
                print(f"⚠️  Errore inizializzazione CSV RAG: {e}")
                print("   Fallback a query pandas")
                self.use_rag = False
                self.rag_indexer = None
        else:
            self.rag_indexer = None
            print("✅ Statistics Agent (pandas queries)")
    
    def _load_csv(self):
        """Carica CSV per fallback"""
        try:
            if os.path.exists(self.csv_path):
                self.df = pd.read_csv(self.csv_path)
                print(f"   - {len(self.df)} righe CSV")
            else:
                print(f"❌ CSV non trovato: {self.csv_path}")
                self.df = pd.DataFrame()
        except Exception as e:
            print(f"❌ Errore caricamento CSV: {e}")
            self.df = pd.DataFrame()
    
    def query_statistics(
        self,
        user_query: str,
        conversation_history: List[Dict] = None
    ) -> Dict:
        """
        Query statistica conversazionale
        
        L'LLM può:
        - Fare domande per chiarire
        - Usare tool per interrogare CSV quando ha confidenza
        - Dare risposte naturali
        
        Args:
            user_query: Query utente
            conversation_history: Storia conversazione
        
        Returns:
            Dict con risposta (può essere domanda o risultato finale)
        """
        
        if self.df is None or self.df.empty:
            return {
                'risposta': 'CSV non disponibile per analisi statistiche.',
                'needs_clarification': False,
                'tool_calls': [],
                'dati': None
            }
        
        # Tool functions disponibili per l'LLM
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_csv_semantic",
                    "description": """Cerca nel CSV usando query semantica naturale. Ritorna TUTTE le righe rilevanti (max 200).
                    
Il CSV contiene queste colonne:
- Prodotto, Descrizione_Prodotto (nome prodotto)
- Cliente, Descrizione_Cliente (nome cliente)
- Esercizio (anno), Periodo (mese 1-12)
- Kg_Venduti_Predetti (kg venduti)
- Ricavo_Netto (€)
- Sconto_Merce, Sconto_Canale, Sconto_Valore (€)
- Cluster_Prodotto_Label, Cluster_Cliente_Label

IMPORTANTE: Il sistema recupera automaticamente TUTTE le righe rilevanti per garantire calcoli consistenti.

Puoi fare query complesse tipo:
- "prodotti con sconto merce maggiore di 10 euro"
- "vendite osvego gennaio 2024"
- "clienti nel cluster rank 3"
- "ricavi netti maggiori di 100 euro"
""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query semantica naturale (es. 'vendite Osvego 3500 anno 2024')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_aggregate",
                    "description": "Calcola aggregazione (somma, media, count) su risultati recuperati tramite search_csv_semantic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["sum", "mean", "median", "count", "min", "max"],
                                "description": "Operazione di aggregazione"
                            },
                            "field": {
                                "type": "string",
                                "enum": ["kg_venduti", "ricavo_netto", "sconto_merce"],
                                "description": "Campo su cui calcolare (default: kg_venduti)"
                            }
                        },
                        "required": ["operation"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_clarification",
                    "description": "Fai una domanda all'utente per chiarire la richiesta. USA quando NON sei sicuro di prodotto/cliente/periodo.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Domanda da fare all'utente"
                            },
                            "suggestions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Opzionale: suggerimenti per l'utente"
                            }
                        },
                        "required": ["question"]
                    }
                }
            }
        ]
        
        # System prompt
        system_prompt = f"""Sei un assistente conversazionale per analisi vendite.

HAI ACCESSO COMPLETO AL CSV con {len(self.df) if self.df is not None else 0} righe tramite retrieval semantico (RAG).

COLONNE DISPONIBILI:
- Prodotto, Descrizione_Prodotto (nome)
- Cliente, Descrizione_Cliente (nome)
- Esercizio (anno), Periodo (mese 1-12)
- Kg_Venduti_Predetti (kg venduti)
- Ricavo_Netto (ricavi in €)
- Sconto_Merce, Sconto_Canale, Sconto_Valore (sconti in €)
- Cluster_Prodotto_Label, Cluster_Cliente_Label

COMPORTAMENTO:

1. **PRIMA DI TUTTO**: Valuta se hai TUTTE le informazioni necessarie
   - Se manca qualcosa o sei incerto: USA ask_clarification

2. **QUANDO HAI CONFIDENZA**:
   - USA search_csv_semantic con query naturale
   - **CRITICO: Per TOTALI/MEDIE/SOMME:**
     - Se search_csv_semantic ritorna summary.kg_totale → USALO direttamente
     - Oppure chiama calculate_aggregate per calcolare su metadata
   - NON guardare solo i documents (sono solo 10 esempi)
   - I calcoli vanno fatti su TUTTI i metadata (fino a 200 righe)

3. **QUERY COMPLESSE** che puoi fare:
   - "prodotti con sconto merce > 10 euro"
   - "clienti nel cluster rank 3"
   - "ricavi netti maggiori di 100 euro nel 2024"
   - "vendite con alto sconto canale"

REGOLE CRITICHE:
- NON fare assunzioni su prodotti/periodi/clienti
- CHIEDI SEMPRE se c'è ambiguità
- USA I TOOL solo quando sei SICURO al 100%
- NON inventare MAI numeri - usa SOLO dati dal CSV
- Le query semantiche trovano automaticamente righe rilevanti

ESEMPI:

Query: "vendite osvego"
Risposta: [USA ask_clarification] "Ho trovato 2 prodotti Osvego nel sistema:
- Osvego 500g
- Osvego 3500 - 14pz x 250g
Quale ti interessa? E per quale periodo?"

Query: "vendite Osvego 3500 nel 2024"
Risposta: [USA search_csv_semantic("vendite Osvego 3500 anno 2024")] 
poi [USA calculate_aggregate("sum")] 
poi rispondi con totale

Query: "prodotti con ricavo netto alto nel Q1"
Risposta: [USA search_csv_semantic("prodotti ricavo netto alto Q1 2024")]
poi analizza risultati e rispondi
"""

        # Messages
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history[-6:])
        
        messages.append({"role": "user", "content": user_query})
        
        # Conversation loop
        max_iterations = 5
        iteration = 0
        tool_calls_log = []
        last_data = None
        needs_clarification = False
        
        while iteration < max_iterations:
            iteration += 1
            
            print(f"\n🔄 Iterazione {iteration}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=0.0  # ← ZERO per determinismo completo
            )
            
            message = response.choices[0].message
            
            # Se non ci sono tool calls, l'LLM ha finito
            if not message.tool_calls:
                risposta_finale = message.content
                print(f"✅ Risposta: {risposta_finale[:100]}...")
                
                return {
                    'risposta': risposta_finale,
                    'needs_clarification': needs_clarification,
                    'tool_calls': tool_calls_log,
                    'dati': last_data,
                    'iterations': iteration
                }
            
            # Esegui tool calls
            messages.append(message)
            
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                print(f"   🔧 Tool: {function_name}")
                print(f"   📋 Args: {json.dumps(arguments, ensure_ascii=False)[:200]}")
                
                # Esegui funzione
                if function_name == "search_csv_semantic":
                    result = self._search_csv_semantic(**arguments)
                    last_data = result
                    
                elif function_name == "calculate_aggregate":
                    if not last_data:
                        result = {"error": "Nessun dato disponibile. Usa prima search_csv_semantic."}
                    else:
                        result = self._calculate_aggregate(last_data, **arguments)
                
                elif function_name == "ask_clarification":
                    # LLM sta chiedendo chiarimenti!
                    needs_clarification = True
                    question = arguments.get('question', '')
                    suggestions = arguments.get('suggestions', [])
                    
                    result = {
                        'clarification_needed': True,
                        'question': question,
                        'suggestions': suggestions
                    }
                    
                    print(f"   ❓ Chiarimento richiesto: {question}")
                    
                    # Ritorna immediatamente con la domanda
                    return {
                        'risposta': question + (f"\n\nOpzioni: {', '.join(suggestions)}" if suggestions else ""),
                        'needs_clarification': True,
                        'question': question,
                        'suggestions': suggestions,
                        'tool_calls': tool_calls_log,
                        'dati': None,
                        'iterations': iteration
                    }
                
                else:
                    result = {"error": f"Funzione {function_name} non riconosciuta"}
                
                tool_calls_log.append({
                    'function': function_name,
                    'arguments': arguments,
                    'result': result
                })
                
                # Aggiungi result ai messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
        
        # Max iterations
        return {
            'risposta': 'Non sono riuscito a completare l\'analisi.',
            'needs_clarification': False,
            'tool_calls': tool_calls_log,
            'dati': last_data,
            'iterations': iteration
        }
    
    def _search_csv_semantic(
        self,
        query: str
    ) -> Dict:
        """Tool function: search semantico con RAG (SEMPRE 200 risultati per consistenza)"""
        
        if not self.use_rag or not self.rag_indexer:
            return {
                'error': 'RAG non disponibile. Installa: pip install chromadb sentence-transformers',
                'num_results': 0
            }
        
        try:
            # Estrai filtri anno/periodo dalla query (se presenti)
            filters = self._extract_filters_from_query(query)
            
            # Query semantica - FISSO 200 risultati per consistenza matematica
            results = self.rag_indexer.search(
                query=query,
                n_results=200,  # Fisso per garantire sempre stesso dataset
                filters=filters  # Filtri metadata
            )
            
            if not results['metadatas'] or not results['metadatas'][0]:
                return {
                    'num_results': 0,
                    'message': 'Nessun risultato trovato per questa query'
                }
            
            metadatas = results['metadatas'][0]
            documents = results['documents'][0]
            
            # Statistiche sui risultati
            kg_totale = sum(m.get('kg_venduti', 0) for m in metadatas)
            ricavo_totale = sum(m.get('ricavo_netto', 0) for m in metadatas)
            
            return {
                'num_results': len(metadatas),
                'metadatas': metadatas,
                'documents': documents[:10],  # Prime 10 per log
                'filters_applied': filters,  # Mostra quali filtri sono stati applicati
                'summary': {
                    'kg_totale': float(kg_totale),
                    'ricavo_totale': float(ricavo_totale),
                    'prodotti_unici': len(set(m.get('prodotto_nome', '') for m in metadatas)),
                    'clienti_unici': len(set(m.get('cliente_nome', '') for m in metadatas))
                }
            }
            
        except Exception as e:
            return {
                'error': f'Errore search RAG: {str(e)}',
                'num_results': 0
            }
    
    def _extract_filters_from_query(self, query: str) -> Dict:
        """Estrae filtri anno/periodo dalla query per ChromaDB where clause"""
        import re
        
        query_lower = query.lower()
        conditions = []
        
        # Estrai anno (2024, 2025, 2026, etc.)
        year_match = re.search(r'\b(202[0-9])\b', query)
        if year_match:
            conditions.append({"esercizio": {"$eq": int(year_match.group(1))}})
        
        # Estrai mese (gennaio, febbraio, ..., o numeri 1-12)
        mesi = {
            'gennaio': 1, 'febbraio': 2, 'marzo': 3, 'aprile': 4,
            'maggio': 5, 'giugno': 6, 'luglio': 7, 'agosto': 8,
            'settembre': 9, 'ottobre': 10, 'novembre': 11, 'dicembre': 12
        }
        
        for mese_nome, mese_num in mesi.items():
            if mese_nome in query_lower:
                conditions.append({"periodo": {"$eq": mese_num}})
                break
        
        # Costruisci where clause ChromaDB
        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            # Multiple conditions → $and
            return {"$and": conditions}
    
    def _calculate_aggregate(
        self,
        data: Dict,
        operation: str,
        field: str = 'kg_venduti'
    ) -> Dict:
        """Tool function: calcola aggregazione su risultati RAG"""
        
        if not data or 'metadatas' not in data:
            return {'error': 'Nessun dato disponibile. Usa prima search_csv_semantic.'}
        
        metadatas = data['metadatas']
        
        if not metadatas:
            return {'error': 'Nessun risultato da aggregare'}
        
        # Estrai valori
        try:
            values = [m.get(field, 0) for m in metadatas if m.get(field) is not None]
            
            if not values:
                return {'error': f'Campo {field} non trovato nei risultati'}
            
            # Calcola aggregazione
            if operation == 'sum':
                value = float(sum(values))
            elif operation == 'mean':
                value = float(sum(values) / len(values))
            elif operation == 'median':
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                if n % 2 == 0:
                    value = float((sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2)
                else:
                    value = float(sorted_vals[n//2])
            elif operation == 'count':
                value = int(len(values))
            elif operation == 'min':
                value = float(min(values))
            elif operation == 'max':
                value = float(max(values))
            else:
                return {'error': f'Operazione {operation} non supportata'}
            
            return {
                'operation': operation,
                'field': field,
                'value': value,
                'num_rows_analyzed': len(metadatas)
            }
            
        except Exception as e:
            return {'error': f'Errore calcolo: {str(e)}'}


if __name__ == "__main__":
    # Test
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = StatisticsAgent()
    
    # Test query
    result = agent.query_statistics(
        "Quanto ho venduto di Osvego nel Q1 2024?"
    )
    
    print("\n=== RISULTATO ===")
    print(f"Risposta: {result['risposta']}")
    print(f"Iterazioni: {result.get('iterations', 0)}")
    print(f"Tool calls: {len(result['tool_calls'])}")
