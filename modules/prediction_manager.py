"""
Modulo 2: Prediction Manager
Gestisce il flusso completo: cerca predizioni nel CSV, genera nuove predizioni se necessario,
e crea risposte in linguaggio naturale usando GPT
"""

import os
import sys
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL_RESPONSE,
    PREDICTIONS_CSV,
    CACHE_PREDICTIONS
)
from modules.predictor import MLPredictor
from modules.trends_analyzer import TrendsAnalyzer


class PredictionManager:
    """
    Manager centrale per gestire predizioni e generare risposte
    """

    def __init__(
        self,
        csv_path: str = str(PREDICTIONS_CSV),
        api_key: str = OPENAI_API_KEY
    ):
        """
        Inizializza il manager

        Args:
            csv_path: Path al CSV delle predizioni
            api_key: Chiave API OpenAI
        """
        self.csv_path = csv_path
        self.df_predictions = None
        self.predictor = MLPredictor()
        self.client = OpenAI(api_key=api_key)
        self.model = OPENAI_MODEL_RESPONSE

        # Carica il CSV se esiste
        self._load_predictions_csv()

        # Inizializza Trends Analyzer
        self.trends_analyzer = TrendsAnalyzer(self.df_predictions, api_key)

    def _load_predictions_csv(self):
        """Carica il CSV delle predizioni"""
        try:
            if os.path.exists(self.csv_path):
                self.df_predictions = pd.read_csv(self.csv_path)
                print(f"✅ CSV predizioni caricato: {len(self.df_predictions)} righe")
            else:
                print(f"⚠️ CSV predizioni non trovato: {self.csv_path}")
                print("Verrà creato al primo salvataggio")
                self.df_predictions = pd.DataFrame()
        except Exception as e:
            print(f"❌ Errore caricamento CSV: {e}")
            self.df_predictions = pd.DataFrame()

    def process_query_combinations(
        self,
        query_combinations: List[Dict],
        use_trends: bool = True
    ) -> Dict:
        """
        Processa le query combinations del Modulo 1

        Args:
            query_combinations: Lista di dict con {prodotto, cliente, periodo}
            use_trends: Se True, analizza anche Google Trends

        Returns:
            Dict con risultati e risposta in linguaggio naturale
        """
        risultati = []

        for combo in query_combinations:
            prodotto = str(combo['prodotto'])
            cliente = str(combo['cliente'])
            mese = int(combo['periodo']['mese'])
            anno = int(combo['periodo']['anno'])

            # Cerca nel CSV o genera nuova predizione
            predizione, fonte = self._get_or_create_prediction(
                prodotto, cliente, anno, mese
            )

            if predizione:
                # Pulisci NaN dal dict predizione
                predizione_clean = self._clean_nan(predizione)

                risultati.append({
                    'prodotto': prodotto,
                    'cliente': cliente,
                    'periodo': {'mese': mese, 'anno': anno},
                    'predizione': predizione_clean,
                    'fonte': fonte  # 'csv' o 'ml_model'
                })
            else:
                risultati.append({
                    'prodotto': prodotto,
                    'cliente': cliente,
                    'periodo': {'mese': mese, 'anno': anno},
                    'errore': 'Predizione non disponibile'
                })

        # Analizza Google Trends se richiesto
        trends_data = None
        if use_trends and len(query_combinations) > 0:
            # Estrai prodotti unici e periodo
            prodotti_unici = list(set([str(c['prodotto']) for c in query_combinations]))
            periodo = query_combinations[0]['periodo']  # Usa il primo periodo

            print(f"\n🔍 Analizzando Google Trends per {len(prodotti_unici)} prodotto/i...")
            try:
                trends_data = self.trends_analyzer.analyze_products_trends(
                    prodotti_unici,
                    periodo
                )
            except Exception as e:
                print(f"⚠️  Errore analisi trends: {e}")
                trends_data = None

        # Genera risposta in linguaggio naturale (con trends se disponibili)
        risposta_naturale = self._generate_natural_response(risultati, trends_data)

        return {
            'risultati': risultati,
            'num_predizioni': len(risultati),
            'trends_data': trends_data,
            'risposta': risposta_naturale,
            'timestamp': datetime.now().isoformat()
        }

    def _clean_nan(self, obj):
        """
        Rimuove NaN da dict/list ricorsivamente, sostituendo con None

        Args:
            obj: Dict, list o valore da pulire

        Returns:
            Oggetto pulito senza NaN
        """
        import math

        if isinstance(obj, dict):
            return {k: self._clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_nan(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        else:
            return obj

    def _get_or_create_prediction(
        self,
        prodotto: str,
        cliente: str,
        anno: int,
        mese: int
    ) -> Tuple[Optional[Dict], str]:
        """
        Cerca la predizione nel CSV, altrimenti la genera con il modello ML

        Args:
            prodotto: Codice prodotto
            cliente: Codice cliente
            anno: Anno
            mese: Mese

        Returns:
            Tuple (predizione_dict, fonte)
        """
        # 1. Cerca nel CSV
        predizione_csv = self._search_in_csv(prodotto, cliente, anno, mese)

        if predizione_csv is not None:
            return predizione_csv, 'csv'

        # 2. Se non trovata, genera con ML
        print(f"📊 Generando nuova predizione per {prodotto}-{cliente}-{mese}/{anno}")

        if not self.predictor.is_ready():
            return None, 'error'

        predizione_ml, errore = self.predictor.predict(
            prodotto=prodotto,
            cliente=cliente,
            anno=anno,
            mese=mese,
            df_storico=self.df_predictions if not self.df_predictions.empty else None
        )

        if predizione_ml:
            # 3. Salva la nuova predizione nel CSV
            self._save_prediction_to_csv(predizione_ml)
            return predizione_ml, 'ml_model'
        else:
            print(f"❌ Errore generazione predizione: {errore}")
            return None, 'error'

    def _search_in_csv(
        self,
        prodotto: str,
        cliente: str,
        anno: int,
        mese: int
    ) -> Optional[Dict]:
        """
        Cerca una predizione nel CSV

        Args:
            prodotto: Codice prodotto
            cliente: Codice cliente
            anno: Anno
            mese: Mese

        Returns:
            Dict con la predizione o None se non trovata
        """
        if self.df_predictions.empty:
            return None

        try:
            # Converte i tipi per il matching
            prodotto_match = int(prodotto) if prodotto.isdigit() else prodotto
            cliente_match = int(cliente) if cliente.isdigit() else cliente

            # Filtra il DataFrame
            mask = (
                (self.df_predictions['Prodotto'] == prodotto_match) &
                (self.df_predictions['Cliente'] == cliente_match) &
                (self.df_predictions['Esercizio'] == anno) &
                (self.df_predictions['Periodo'] == mese)
            )

            risultato = self.df_predictions[mask]

            if len(risultato) > 0:
                row = risultato.iloc[0]

                # Estrai valori con gestione NaN
                kg_predetti = row.get('Kg_Venduti_Predetti')
                if pd.notna(kg_predetti):
                    kg_predetti = float(kg_predetti)
                else:
                    kg_predetti = None

                desc_prodotto = row.get('Descrizione_Prodotto', 'N/A')
                if pd.isna(desc_prodotto):
                    desc_prodotto = 'N/A'

                desc_cliente = row.get('Descrizione_Cliente', 'N/A')
                if pd.isna(desc_cliente):
                    desc_cliente = 'N/A'

                kg_reali = None
                if 'Kg_Venduti_Reali' in row and pd.notna(row['Kg_Venduti_Reali']):
                    kg_reali = float(row['Kg_Venduti_Reali'])

                return {
                    'prodotto': str(row['Prodotto']),
                    'cliente': str(row['Cliente']),
                    'anno': int(row['Esercizio']),
                    'mese': int(row['Periodo']),
                    'kg_predetti': kg_predetti,
                    'descrizione_prodotto': desc_prodotto,
                    'descrizione_cliente': desc_cliente,
                    'kg_reali': kg_reali,
                    'fonte': 'csv'
                }

            return None

        except Exception as e:
            print(f"Errore ricerca CSV: {e}")
            return None

    def _save_prediction_to_csv(self, predizione: Dict):
        """
        Salva una nuova predizione nel CSV

        Args:
            predizione: Dict con i dati della predizione
        """
        try:
            # Crea una nuova riga
            nuova_riga = {
                'Esercizio': predizione['anno'],
                'Periodo': predizione['mese'],
                'Data': f"{predizione['anno']}-{predizione['mese']:02d}-01",
                'Prodotto': predizione['prodotto'],
                'Descrizione_Prodotto': 'N/A',  # Potrebbe essere popolato da un database
                'Cliente': predizione['cliente'],
                'Descrizione_Cliente': 'N/A',
                'Kg_Venduti_Predetti': predizione['kg_predetti'],
                'Kg_Venduti_Reali': None,
                'Tipo_Periodo': 'Predetto_Runtime',
                'Data_Predizione': predizione.get('data_predizione', datetime.now().isoformat()),
                'Modello': predizione.get('modello', 'GradientBoosting'),
                'Confidenza': predizione.get('confidenza', 0.7)
            }

            # Aggiungi al DataFrame in memoria
            if self.df_predictions.empty:
                self.df_predictions = pd.DataFrame([nuova_riga])
            else:
                self.df_predictions = pd.concat(
                    [self.df_predictions, pd.DataFrame([nuova_riga])],
                    ignore_index=True
                )

            # Salva su disco
            self.df_predictions.to_csv(self.csv_path, index=False)
            print(f"💾 Predizione salvata nel CSV: {self.csv_path}")

        except Exception as e:
            print(f"❌ Errore salvataggio CSV: {e}")

    def _generate_natural_response(self, risultati: List[Dict], trends_data: Optional[Dict] = None) -> str:
        """
        Genera una risposta in linguaggio naturale usando GPT

        Args:
            risultati: Lista di risultati delle predizioni
            trends_data: Dati Google Trends (opzionale)

        Returns:
            Risposta in linguaggio naturale
        """
        try:
            # Prepara il contesto per GPT
            contesto_predizioni = self._prepare_context_for_gpt(risultati)

            # Prepara contesto trends se disponibile
            contesto_trends = ""
            if trends_data:
                contesto_trends = self.trends_analyzer.format_trends_for_llm(trends_data)

            system_prompt = """Sei un assistente esperto di demand forecasting per Gentilini, azienda italiana di biscotti premium.

Il tuo compito è presentare le predizioni di vendita in modo chiaro e professionale,
integrando anche fattori esogeni da Google Trends quando disponibili.

Linee guida:
- Usa un tono professionale ma friendly
- Presenta i dati in modo strutturato e leggibile
- Evidenzia informazioni chiave (kg predetti, periodo)
- Se ci sono dati Google Trends, integra insights su tendenze di mercato e stagionalità
- Collega i trends alle predizioni quando pertinente
- Se una predizione viene dal CSV (dati storici), menzionalo
- Se una predizione è stata generata al momento, menzionalo
- Aggiungi insights strategici quando appropriato
- Usa emoji occasionalmente per rendere più leggibile (📊 📈 🔍 ✅)
"""

            # Costruisci prompt utente
            user_parts = [f"""Ecco i risultati delle predizioni richieste:

{contesto_predizioni}"""]

            if contesto_trends:
                user_parts.append(f"\n{contesto_trends}")
                user_parts.append("\nPer favore, genera una risposta completa che integri sia le predizioni che i fattori esogeni da Google Trends, fornendo insights utili.")
            else:
                user_parts.append("\nPer favore, genera una risposta chiara e professionale che presenti questi dati all'utente.")

            user_prompt = "\n".join(user_parts)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Errore generazione risposta: {e}")
            # Fallback a una risposta semplice
            return self._generate_simple_response(risultati)

    def _prepare_context_for_gpt(self, risultati: List[Dict]) -> str:
        """Prepara il contesto testuale per GPT"""
        context_parts = []

        for i, ris in enumerate(risultati, 1):
            if 'errore' in ris:
                context_parts.append(
                    f"{i}. Prodotto {ris['prodotto']}, Cliente {ris['cliente']}, "
                    f"{ris['periodo']['mese']}/{ris['periodo']['anno']}: ERRORE - {ris['errore']}"
                )
            else:
                pred = ris['predizione']
                fonte_desc = "dati storici (CSV)" if ris['fonte'] == 'csv' else "modello ML (generato ora)"

                context_parts.append(
                    f"{i}. Prodotto {ris['prodotto']}, Cliente {ris['cliente']}, "
                    f"Periodo: {ris['periodo']['mese']}/{ris['periodo']['anno']}\n"
                    f"   - Kg predetti: {pred.get('kg_predetti', 0):.2f}\n"
                    f"   - Fonte: {fonte_desc}\n"
                    f"   - Confidenza: {pred.get('confidenza', 0.7):.0%}"
                )

        return "\n\n".join(context_parts)

    def _generate_simple_response(self, risultati: List[Dict]) -> str:
        """Genera una risposta semplice senza GPT (fallback)"""
        response_parts = ["📊 Ecco le predizioni richieste:\n"]

        for i, ris in enumerate(risultati, 1):
            if 'errore' in ris:
                response_parts.append(
                    f"{i}. ❌ Prodotto {ris['prodotto']}, Cliente {ris['cliente']}: {ris['errore']}"
                )
            else:
                pred = ris['predizione']
                kg = pred.get('kg_predetti', 0)
                fonte = "📁 CSV" if ris['fonte'] == 'csv' else "🤖 ML"

                response_parts.append(
                    f"{i}. {fonte} Prodotto {ris['prodotto']}, Cliente {ris['cliente']}, "
                    f"{ris['periodo']['mese']}/{ris['periodo']['anno']}: {kg:.2f} kg"
                )

        return "\n".join(response_parts)


if __name__ == "__main__":
    # Test del manager
    from dotenv import load_dotenv
    load_dotenv()

    manager = PredictionManager()

    # Test con query combinations di esempio
    test_combinations = [
        {
            'prodotto': '40000',
            'cliente': '393',
            'periodo': {'mese': 1, 'anno': 2024}
        },
        {
            'prodotto': '40000',
            'cliente': '439',
            'periodo': {'mese': 1, 'anno': 2024}
        }
    ]

    print("\n=== TEST DEL PREDICTION MANAGER ===\n")
    risultato = manager.process_query_combinations(test_combinations)

    print(f"\nNumero predizioni: {risultato['num_predizioni']}")
    print(f"\nRisposta:\n{risultato['risposta']}")
