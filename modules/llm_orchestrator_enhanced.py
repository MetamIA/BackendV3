"""
LLM Orchestrator - Entity Extraction First
GPT estrae prodotti/clienti/periodo → poi NameCodeMapper traduce
"""

import os
import sys
import json
from typing import Dict, List, Optional
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import OPENAI_API_KEY, OPENAI_MODEL_QUERY


class LLMOrchestratorEnhanced:
    """
    Orchestratore con entity extraction GPT
    """
    
    def __init__(self, api_key: str = OPENAI_API_KEY):
        """Inizializza orchestratore"""
        self.client = OpenAI(api_key=api_key)
        self.model = OPENAI_MODEL_QUERY
    
    def analyze_and_plan(
        self, 
        user_message: str,
        conversation_history: List[Dict] = None
    ) -> Dict:
        """
        Analizza query e decide piano
        """
        
        system_prompt = """Sei un orchestratore per un sistema di demand forecasting.

Il tuo compito è ESTRARRE le entità dalla query e decidere l'operazione.

REGOLA PRINCIPALE: 
- Se la query chiede DATI (vendite, previsioni, totali, medie, trends, analisi) → tipo_richiesta: "prediction"
- Solo se è una domanda generica (saluto, help, ecc.) → tipo_richiesta: "chat"

TIPI DI RICHIESTA:

1. **Dati numerici** (vendite, previsioni, somme, medie):
   → tipo_richiesta: "prediction"
   → operazione: predizione_singola|somma|media|mediana|aggregazione_clienti

2. **Analisi trends** (analisi, trend, crescita, pattern, stagionalità):
   → tipo_richiesta: "prediction"  
   → operazione: "trends_analysis"
   → input: {prodotti, clienti (opzionale), periodo_range}

3. **Chat generica**:
   → tipo_richiesta: "chat"

ENTITÀ DA ESTRARRE:

1. **prodotti** - Lista nomi prodotti menzionati (anche parziali)
   - Esempi: ["Osvego 3500"], ["Gran Panettone"], ["Strenna Natale"]
   - Se NON menzionato → ["*"] (tutti i prodotti)

2. **clienti** - Lista nomi clienti menzionati
   - Esempi: ["MATTEONI"], ["RUGGERI ALVARO"], ["ELLO FOOD"]
   - Se NON menzionato → [] (vuoto)
   - IMPORTANTE: "vendite", "totale", "somma" NON sono clienti!

3. **periodo** - Periodo temporale
   - Singolo: {mese: 1, anno: 2024}
   - Range: {da: {mese: 1, anno: 2024}, a: {mese: 3, anno: 2024}}
   - Se NON specificato → anno corrente 2024

OPERAZIONI:

a) **"predizione_singola"** - Predizione per UN periodo specifico
   Usa quando: periodo singolo specificato (es. "gennaio 2024")

b) **"somma"** - Somma kg in range
   Usa quando: "totale", "somma", "Q1", "primo semestre", range periodi

c) **"media"** - Media kg in range
   Usa quando: "media", "media mensile", "mediamente"

d) **"mediana"** - Mediana kg
   Usa quando: "mediana"

e) **"aggregazione_clienti"** - Aggregazione su tutti i clienti
   Usa quando: clienti = [] (vuoto, non specificato)

f) **"trends_analysis"** - Analisi trends e pattern
   Usa quando: "analisi", "trends", "trend", "crescita", "pattern", "stagionalità", "andamento"

g) **"rag_query"** - Query su documenti aziendali
   Usa quando: "cerca documento", "policy", "procedura", "regolamento", "manuale", "trova informazioni su"

REGOLE CRITICHE:

1. **Se clienti NON menzionato → clienti: []**
   - "Quanto venduto prodotto X?" → clienti: []
   - "Totale vendite Osvego nel Q1" → clienti: []

2. **Parole comuni NON sono clienti:**
   - "vendite", "venduto", "totale", "somma" → NON clienti!

3. **Estrai nomi parziali prodotti:**
   - "Osvego 3500" → prodotti: ["Osvego 3500"]
   - "Gran Panettone" → prodotti: ["Gran Panettone"]

4. **Se periodo NON specificato → usa anno corrente 2024**

ESEMPI:

Query: "Quanto ho venduto di Osvego 3500 a gennaio 2024?"
→ tipo_richiesta: "prediction"  ← Chiede DATI!
→ prodotti: ["Osvego 3500"]
→ clienti: []  ← NON specificato!
→ periodo: {mese: 1, anno: 2024}
→ operazione: "aggregazione_clienti" (clienti vuoto)
→ attivo: true

Query: "Totale vendite Osvego 3500 nel Q1 2024"
→ tipo_richiesta: "prediction"  ← Chiede DATI!
→ prodotti: ["Osvego 3500"]
→ clienti: []  ← "vendite" NON è un cliente!
→ periodo: {da: {mese: 1, anno: 2024}, a: {mese: 3, anno: 2024}}
→ operazione: "somma"
→ attivo: true

Query: "Previsioni Osvego per cliente MATTEONI febbraio 2025"
→ tipo_richiesta: "prediction"  ← Chiede DATI!
→ prodotti: ["Osvego"]
→ clienti: ["MATTEONI"]  ← Cliente specificato!
→ periodo: {mese: 2, anno: 2025}
→ operazione: "predizione_singola"
→ attivo: true

Query: "Analisi trends Filone Fette Tostate"
→ tipo_richiesta: "prediction"  ← Chiede ANALISI!
→ prodotti: ["Filone Fette Tostate"]
→ clienti: []  ← Non specificato
→ periodo_range: {da: {mese: 1, anno: 2023}, a: {mese: 12, anno: 2024}}
→ operazione: "trends_analysis"
→ attivo: true

Query: "Quali sono le policy aziendali sui resi?"
→ tipo_richiesta: "prediction"  ← Chiede documenti!
→ operazione: "rag_query"
→ input: {question: "policy aziendali resi"}
→ attivo: true

Query: "Ciao, come funziona il sistema?"
→ tipo_richiesta: "chat"  ← NON chiede dati
→ risposta_diretta: "Spiegazione..."

Rispondi in JSON:

Per richieste DATI:
{
  "tipo_richiesta": "prediction",
  "entita": {
    "prodotti": [...],
    "clienti": [...],
    "periodo_estratto": "..."
  },
  "moduli": {
    "prediction": {
      "attivo": true,
      "operazione": "predizione_singola|somma|media|mediana|aggregazione_clienti",
      "input": {
        "prodotti": [...],
        "clienti": [...] o "*",
        "periodo": {...} o "periodo_range": {...}
      }
    }
  }
}

Per richieste CHAT:
{
  "tipo_richiesta": "chat",
  "risposta_diretta": "..."
}"""

        context_messages = []
        if conversation_history:
            context_messages = conversation_history[-8:]
        
        user_prompt = f"""Richiesta utente: "{user_message}"

Data corrente: Dicembre 2024
Anno corrente: 2024

Analizza e genera piano JSON."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt}
                ] + context_messages + [
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Molto bassa per massimo determinismo
                response_format={"type": "json_object"}
            )
            
            raw_response = response.choices[0].message.content
            print(f"\n🤖 GPT Raw Response:\n{raw_response}\n")
            
            plan = json.loads(raw_response)
            plan = self._validate_plan(plan)
            
            print(f"\n🧠 Piano LLM Orchestrator:")
            print(f"   Tipo: {plan['tipo_richiesta']}")
            
            if plan.get('entita'):
                print(f"   Entità estratte:")
                print(f"      Prodotti: {plan['entita'].get('prodotti', [])}")
                print(f"      Clienti: {plan['entita'].get('clienti', [])}")
            
            if plan['moduli'].get('prediction', {}).get('attivo'):
                pred = plan['moduli']['prediction']
                print(f"   Operazione: {pred.get('operazione', 'N/A')}")
                print(f"   Clienti input: {pred.get('input', {}).get('clienti', [])}")
            
            return plan
            
        except Exception as e:
            print(f"❌ Errore orchestrator: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_plan()
    
    def _validate_plan(self, plan: Dict) -> Dict:
        """Valida piano"""
        if 'tipo_richiesta' not in plan:
            plan['tipo_richiesta'] = 'chat'
        
        if 'moduli' not in plan:
            plan['moduli'] = {}
        
        if 'prediction' in plan['moduli'] and plan['moduli']['prediction'].get('attivo'):
            pred = plan['moduli']['prediction']
            
            if 'operazione' not in pred:
                pred['operazione'] = 'predizione_singola'
            
            if 'input' not in pred:
                pred['input'] = {}
            
            # Se clienti vuoto → usa "*" MA NON cambiare operazione!
            # GPT ha già deciso l'operazione corretta (somma, media, ecc.)
            if 'clienti' not in pred['input'] or not pred['input']['clienti'] or pred['input']['clienti'] == []:
                pred['input']['clienti'] = '*'
        
        return plan
    
    def _fallback_plan(self) -> Dict:
        """Piano fallback"""
        return {
            'tipo_richiesta': 'chat',
            'moduli': {
                'prediction': {'attivo': False}
            },
            'risposta_diretta': "Mi dispiace, non ho capito. Puoi riformulare?"
        }
