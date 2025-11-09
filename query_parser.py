"""
Modulo per il parsing delle query di demand forecasting usando GPT
"""

import os
from typing import Dict, List, Optional, Union
import json
from openai import OpenAI


class QueryParser:
    """
    Classe per analizzare le frasi dell'utente e distinguere tra:
    - Chat normale
    - Query di forecasting (con estrazione di prodotto, cliente, periodo)
    """
    
    def __init__(self, api_key: str):
        """
        Inizializza il parser con la chiave API di OpenAI
        
        Args:
            api_key: Chiave API di OpenAI
        """
        self.client = OpenAI(api_key=api_key)
        
    def _create_system_prompt(self) -> str:
        """Crea il prompt di sistema per GPT"""
        return """Sei un assistente per un sistema di demand forecasting. 
Il tuo compito è analizzare le frasi dell'utente e determinare se si tratta di:
1. Una chat normale (saluti, domande generiche, conversazione)
2. Una query di forecasting (richieste che contengono informazioni su prodotti, clienti e periodi)

Per le QUERY DI FORECASTING, devi estrarre:
- prodotti: lista di nomi di prodotti menzionati
- clienti: lista di nomi di clienti menzionati
- periodo: oggetto con "mese" (numero 1-12 o nome) e "anno" (numero a 4 cifre)

Rispondi SEMPRE in formato JSON con questa struttura:
{
    "tipo": "chat" oppure "query",
    "dati": {
        "prodotti": [...],
        "clienti": [...],
        "periodo": {"mese": ..., "anno": ...}
    },
    "risposta_chat": "testo della risposta se tipo=chat"
}

Se tipo="chat", il campo "dati" può essere null e devi fornire "risposta_chat".
Se tipo="query", devi estrarre prodotti, clienti e periodo. Se mancano informazioni essenziali, imposta tipo="chat" e chiedi chiarimenti.

Esempi:
- "Ciao come stai?" -> tipo: "chat", risposta_chat: "Ciao! Sto bene, grazie. Come posso aiutarti?"
- "Previsioni per prodotto A cliente X a gennaio 2025" -> tipo: "query", dati con prodotti:["A"], clienti:["X"], periodo:{mese:1, anno:2025}
- "Vendite di pasta e riso per cliente Rossi e Bianchi a marzo 2025" -> tipo: "query", prodotti:["pasta","riso"], clienti:["Rossi","Bianchi"]
"""

    def parse_message(self, user_message: str) -> Dict:
        """
        Analizza il messaggio dell'utente e determina se è chat o query
        
        Args:
            user_message: Il messaggio dell'utente
            
        Returns:
            Dict con la struttura della risposta
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Usa gpt-4o-mini per essere economico, o gpt-4o per qualità massima
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # Bassa temperatura per risposte più consistenti
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Se è una query, genera il prodotto cartesiano
            if result.get("tipo") == "query":
                result = self._generate_cartesian_product(result)
            
            return result
            
        except Exception as e:
            return {
                "tipo": "error",
                "errore": str(e),
                "risposta_chat": f"Mi dispiace, si è verificato un errore: {str(e)}"
            }
    
    def _generate_cartesian_product(self, result: Dict) -> Dict:
        """
        Genera il prodotto cartesiano di prodotti x clienti
        
        Args:
            result: Risultato dal parsing GPT
            
        Returns:
            Risultato arricchito con il prodotto cartesiano
        """
        dati = result.get("dati", {})
        prodotti = dati.get("prodotti", [])
        clienti = dati.get("clienti", [])
        periodo = dati.get("periodo", {})
        
        # Genera tutte le combinazioni prodotto x cliente
        combinazioni = []
        for prodotto in prodotti:
            for cliente in clienti:
                combinazioni.append({
                    "prodotto": prodotto,
                    "cliente": cliente,
                    "periodo": periodo
                })
        
        result["query_combinations"] = combinazioni
        result["num_combinations"] = len(combinazioni)
        
        return result


def process_user_input(user_message: str, api_key: str) -> Dict:
    """
    Funzione helper per processare l'input dell'utente
    
    Args:
        user_message: Il messaggio dell'utente
        api_key: Chiave API di OpenAI
        
    Returns:
        Dizionario con il risultato del parsing
    """
    parser = QueryParser(api_key)
    return parser.parse_message(user_message)


if __name__ == "__main__":
    # Test del modulo
    API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    test_messages = [
        "Ciao! Come stai?",
        "Dammi le previsioni per pasta cliente Rossi a gennaio 2025",
        "Vendite di latte e burro per clienti Bianchi e Verdi a marzo 2025",
        "Qual è il meteo di oggi?"
    ]
    
    parser = QueryParser(API_KEY)
    
    print("=== TEST DEL QUERY PARSER ===\n")
    for msg in test_messages:
        print(f"Input: {msg}")
        result = parser.parse_message(msg)
        print(f"Output: {json.dumps(result, indent=2, ensure_ascii=False)}")
        print("-" * 80)
