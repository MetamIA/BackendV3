"""
Statistics Agent - Conversational CSV analysis with tool calling
L'LLM vede prodotti/clienti unici e usa tool per interrogare iterativamente il CSV
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Any
from openai import OpenAI
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import OPENAI_API_KEY, OPENAI_MODEL_QUERY, PREDICTIONS_CSV


class StatisticsAgent:
    """
    Agent conversazionale per statistiche CSV
    L'LLM usa tool functions per interrogare iterativamente il CSV
    """
    
    COL_PRODOTTO = 'Prodotto'  # Codice numerico
    COL_CLIENTE = 'Cliente'    # Codice numerico
    COL_PERIODO = 'Periodo'
    COL_ESERCIZIO = 'Esercizio'
    COL_KG = 'Kg_Venduti_Predetti'
    COL_DESC_PRODOTTO = 'Descrizione_Prodotto'  # Nome leggibile
    COL_DESC_CLIENTE = 'Descrizione_Cliente'    # Nome leggibile
    
    def __init__(
        self,
        csv_path: str = str(PREDICTIONS_CSV),
        api_key: str = OPENAI_API_KEY
    ):
        """Inizializza agent"""
        self.csv_path = csv_path
        self.client = OpenAI(api_key=api_key)
        self.model = OPENAI_MODEL_QUERY  # gpt-4o-mini
        
        # Carica CSV
        self.df = None
        self.unique_products = []
        self.unique_clients = []
        self._load_csv()
    
    def _load_csv(self):
        """Carica CSV ed estrae valori unici"""
        try:
            if os.path.exists(self.csv_path):
                self.df = pd.read_csv(self.csv_path)
                
                # Estrai NOMI leggibili (non codici)
                if self.COL_DESC_PRODOTTO in self.df.columns:
                    self.unique_products = sorted(self.df[self.COL_DESC_PRODOTTO].dropna().unique().tolist())
                else:
                    print(f"   ⚠️  Colonna {self.COL_DESC_PRODOTTO} non trovata")
                    self.unique_products = []
                
                if self.COL_DESC_CLIENTE in self.df.columns:
                    self.unique_clients = sorted(self.df[self.COL_DESC_CLIENTE].dropna().unique().tolist())
                else:
                    print(f"   ⚠️  Colonna {self.COL_DESC_CLIENTE} non trovata")
                    self.unique_clients = []
                
                print(f"✅ Statistics Agent inizializzato")
                print(f"   - {len(self.df)} righe CSV")
                print(f"   - {len(self.unique_products)} prodotti unici")
                print(f"   - {len(self.unique_clients)} clienti unici")
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
                    "name": "query_csv_data",
                    "description": "Interroga il CSV per recuperare dati filtrati. USA SOLO quando sei SICURO di prodotto/cliente/periodo. Se non sei sicuro, CHIEDI chiarimenti all'utente.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prodotto_nome_contiene": {
                                "type": "string",
                                "description": "Cerca prodotti il cui nome CONTIENE questa stringa (es. 'Osvego')"
                            },
                            "cliente_nome_contiene": {
                                "type": "string",
                                "description": "Cerca clienti il cui nome CONTIENE questa stringa"
                            },
                            "periodo_mese": {
                                "type": "integer",
                                "description": "Mese specifico (1-12). USA SOLO se utente ha specificato mese."
                            },
                            "periodo_anno": {
                                "type": "integer",
                                "description": "Anno. USA SOLO se utente ha specificato anno o è implicito."
                            },
                            "periodo_da_mese": {
                                "type": "integer",
                                "description": "Range: mese inizio (1-12)"
                            },
                            "periodo_da_anno": {
                                "type": "integer",
                                "description": "Range: anno inizio"
                            },
                            "periodo_a_mese": {
                                "type": "integer",
                                "description": "Range: mese fine (1-12)"
                            },
                            "periodo_a_anno": {
                                "type": "integer",
                                "description": "Range: anno fine"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Massimo numero righe da ritornare (default 100, max 500)"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_aggregate",
                    "description": "Calcola aggregazione (somma, media, count) su dati recuperati. Calcola sempre su colonna Kg_Venduti_Predetti.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["sum", "mean", "median", "count", "min", "max"],
                                "description": "Operazione di aggregazione da calcolare sui kg venduti"
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
                                "description": "Opzionale: suggerimenti per l'utente (es. lista prodotti simili)"
                            }
                        },
                        "required": ["question"]
                    }
                }
            }
        ]
        
        # System prompt con prodotti/clienti
        system_prompt = f"""Sei un assistente conversazionale per analisi vendite.

HAI ACCESSO A:
- CSV con {len(self.df)} righe di vendite
- {len(self.unique_products)} prodotti unici
- {len(self.unique_clients)} clienti unici

PRODOTTI DISPONIBILI (campione):
{json.dumps(self.unique_products[:30], ensure_ascii=False)}

CLIENTI DISPONIBILI (campione):
{json.dumps(self.unique_clients[:30], ensure_ascii=False)}

COMPORTAMENTO:

1. **PRIMA DI TUTTO**: Valuta se hai TUTTE le informazioni necessarie
   - Prodotto: chiaro? Ci sono varianti (es. "Osvego 500g" vs "Osvego 3500")?
   - Periodo: specificato? "2024" significa tutto l'anno o un mese?
   - Cliente: specificato o tutti?

2. **SE MANCA QUALCOSA O SEI INCERTO**:
   - USA ask_clarification per chiedere all'utente
   - Fornisci suggerimenti basati su prodotti/clienti disponibili
   - Esempio: "Ho trovato 2 prodotti Osvego: 'Osvego 500g' e 'Osvego 3500'. Quale ti interessa?"

3. **SOLO QUANDO HAI CONFIDENZA**:
   - USA query_csv_data per recuperare dati
   - USA calculate_aggregate per calcolare statistiche
   - Dai risposta finale

REGOLE CRITICHE:
- NON fare assunzioni su prodotti/periodi/clienti
- CHIEDI SEMPRE se c'è ambiguità
- USA I TOOL solo quando sei SICURO al 100%
- NON inventare MAI numeri - usa SOLO dati dal CSV

ESEMPI:

Query: "vendite osvego"
Risposta: [USA ask_clarification] "Ho trovato 2 prodotti Osvego nel sistema:
- Osvego 500g
- Osvego 3500 - 14pz x 250g
Quale ti interessa? E per quale periodo?"

Query: "vendite Osvego 3500 nel 2024"
Risposta: [USA query_csv_data] poi calcola e rispondi

Query: "totale Q1"
Risposta: [USA ask_clarification] "Per quale prodotto vuoi il totale del Q1 2024?"
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
                temperature=0.2
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
                if function_name == "query_csv_data":
                    result = self._query_csv_data(**arguments)
                    last_data = result
                    
                elif function_name == "calculate_aggregate":
                    if not last_data:
                        result = {"error": "Nessun dato disponibile. Usa prima query_csv_data."}
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
    
    def _query_csv_data(
        self,
        prodotto_codice: str = None,
        prodotto_nome_contiene: str = None,
        cliente_codice: str = None,
        cliente_nome_contiene: str = None,
        periodo_mese: int = None,
        periodo_anno: int = None,
        periodo_da_mese: int = None,
        periodo_da_anno: int = None,
        periodo_a_mese: int = None,
        periodo_a_anno: int = None,
        limit: int = 100
    ) -> Dict:
        """Tool function: query CSV con filtri"""
        
        df = self.df.copy()
        
        # Filtri prodotto
        if prodotto_codice:
            # Cerca per codice esatto
            df = df[df[self.COL_PRODOTTO].astype(str) == str(prodotto_codice)]
        elif prodotto_nome_contiene:
            # Cerca per NOME nella colonna Descrizione_Prodotto
            if self.COL_DESC_PRODOTTO in df.columns:
                df = df[df[self.COL_DESC_PRODOTTO].astype(str).str.contains(prodotto_nome_contiene, case=False, na=False)]
            else:
                # Fallback: cerca nel codice
                df = df[df[self.COL_PRODOTTO].astype(str).str.contains(prodotto_nome_contiene, case=False, na=False)]
        
        # Filtri cliente
        if cliente_codice:
            # Cerca per codice esatto
            df = df[df[self.COL_CLIENTE].astype(str) == str(cliente_codice)]
        elif cliente_nome_contiene:
            # Cerca per NOME nella colonna Descrizione_Cliente
            if self.COL_DESC_CLIENTE in df.columns:
                df = df[df[self.COL_DESC_CLIENTE].astype(str).str.contains(cliente_nome_contiene, case=False, na=False)]
            else:
                # Fallback: cerca nel codice
                df = df[df[self.COL_CLIENTE].astype(str).str.contains(cliente_nome_contiene, case=False, na=False)]
        
        # Filtri periodo
        if periodo_mese and periodo_anno:
            df = df[(df[self.COL_PERIODO] == periodo_mese) & (df[self.COL_ESERCIZIO] == periodo_anno)]
        elif periodo_anno and not periodo_mese:
            # Solo anno specificato → tutto l'anno
            df = df[df[self.COL_ESERCIZIO] == periodo_anno]
        elif periodo_da_mese and periodo_da_anno and periodo_a_mese and periodo_a_anno:
            # Range periodo
            df = df[
                ((df[self.COL_ESERCIZIO] > periodo_da_anno) | 
                 ((df[self.COL_ESERCIZIO] == periodo_da_anno) & (df[self.COL_PERIODO] >= periodo_da_mese))) &
                ((df[self.COL_ESERCIZIO] < periodo_a_anno) | 
                 ((df[self.COL_ESERCIZIO] == periodo_a_anno) & (df[self.COL_PERIODO] <= periodo_a_mese)))
            ]
        
        # Limit
        df = df.head(min(limit, 500))
        
        # Ritorna dati
        if df.empty:
            return {
                'num_rows': 0,
                'message': 'Nessun dato trovato con questi filtri'
            }
        
        return {
            'num_rows': len(df),
            'columns': df.columns.tolist(),
            'data': df.to_dict(orient='records')[:limit],
            'summary': {
                'kg_totali': float(df[self.COL_KG].sum()),
                'kg_media': float(df[self.COL_KG].mean()),
                'num_clienti_unici': int(df[self.COL_CLIENTE].nunique()),
                'num_prodotti_unici': int(df[self.COL_PRODOTTO].nunique())
            }
        }
    
    def _calculate_aggregate(
        self,
        data: Dict,
        operation: str,
        column: str = None
    ) -> Dict:
        """Tool function: calcola aggregazione su dati"""
        
        if not data or 'data' not in data:
            return {'error': 'Nessun dato disponibile. Usa prima query_csv_data.'}
        
        # Default column
        if not column:
            column = self.COL_KG
        
        # Ricostruisci DataFrame
        df = pd.DataFrame(data['data'])
        
        if column not in df.columns:
            return {'error': f'Colonna {column} non trovata'}
        
        # Calcola aggregazione semplice
        if operation == 'sum':
            value = float(df[column].sum())
        elif operation == 'mean':
            value = float(df[column].mean())
        elif operation == 'median':
            value = float(df[column].median())
        elif operation == 'count':
            value = int(df[column].count())
        elif operation == 'min':
            value = float(df[column].min())
        elif operation == 'max':
            value = float(df[column].max())
        else:
            return {'error': f'Operazione {operation} non supportata'}
        
        return {
            'operation': operation,
            'column': column,
            'value': value,
            'num_rows_analyzed': len(df)
        }


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
