"""
Modulo 1: Query Parser
Analizza i messaggi dell'utente e distingue tra chat normale e query di forecasting
"""

import json
from typing import Dict, List, Optional
from openai import OpenAI
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import OPENAI_API_KEY, OPENAI_MODEL_QUERY


class QueryParser:
    """
    Classe per analizzare le frasi dell'utente e distinguere tra:
    - Chat normale
    - Query di forecasting (con estrazione di prodotto, cliente, periodo)
    """
    
    def __init__(self, api_key: str = OPENAI_API_KEY):
        """
        Inizializza il parser con la chiave API di OpenAI
        
        Args:
            api_key: Chiave API di OpenAI
        """
        if not api_key:
            raise ValueError("OPENAI_API_KEY non fornita")
        
        self.client = OpenAI(api_key=api_key)
        self.model = OPENAI_MODEL_QUERY
        
    def _create_system_prompt(self) -> str:
        """Crea il prompt di sistema per GPT"""
        return """Sei un assistente per un sistema di demand forecasting. 
Il tuo compito è analizzare le frasi dell'utente e determinare se si tratta di:
1. Una chat normale (saluti, domande generiche, conversazione)
2. Una query di forecasting (richieste che contengono informazioni su prodotti, clienti e periodi)

Per le QUERY DI FORECASTING, devi estrarre:
- prodotti: lista di codici o nomi di prodotti menzionati
- clienti: lista di codici o nomi di clienti menzionati  
- periodo: oggetto con "mese" (numero 1-12) e "anno" (numero a 4 cifre)

IMPORTANTE:
- I codici prodotto sono numeri di 5 cifre (es: 40000)
- I codici cliente sono numeri di 3-8 cifre (es: 393, 10000018)
- I nomi possono essere usati al posto dei codici
- Il mese può essere scritto come nome o numero

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
Se tipo="query", devi estrarre prodotti, clienti e periodo. Se mancano informazioni essenziali, chiedi chiarimenti.

Esempi:
- "Ciao come stai?" → tipo: "chat"
- "Previsioni per prodotto 40000 cliente 393 a gennaio 2025" → tipo: "query"
- "Vendite di pasta per Rossi a marzo 2025" → tipo: "query" 
- "Quanto venderemo di prodotto 40000 al cliente 10000018 nel mese 6 del 2024?" → tipo: "query"
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
                model=self.model,
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
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


if __name__ == "__main__":
    # Test del modulo
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    API_KEY = os.getenv("OPENAI_API_KEY")
    
    test_messages = [
        "Ciao! Come stai?",
        "Previsioni per prodotto 40000 cliente 393 a gennaio 2024",
        "Vendite di prodotto 40000 per clienti 393 e 439 a marzo 2024",
    ]
    
    parser = QueryParser(API_KEY)
    
    print("=== TEST DEL QUERY PARSER ===\n")
    for msg in test_messages:
        print(f"Input: {msg}")
        result = parser.parse_message(msg)
        print(f"Output: {json.dumps(result, indent=2, ensure_ascii=False)}")
        print("-" * 80)
