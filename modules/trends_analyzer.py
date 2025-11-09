"""
Modulo 3: Trends Analyzer
Analizza Google Trends per fattori esogeni relativi ai prodotti
"""

import os
import sys
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from openai import OpenAI

# PyTrends
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    print("⚠️  pytrends non installato. Installa con: pip install pytrends")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import OPENAI_API_KEY, OPENAI_MODEL_RESPONSE


class TrendsAnalyzer:
    """
    Analizzatore Google Trends per fattori esogeni
    """

    def __init__(
        self,
        df_predictions: pd.DataFrame = None,
        api_key: str = OPENAI_API_KEY
    ):
        """
        Inizializza l'analizzatore

        Args:
            df_predictions: DataFrame con le predizioni (include descrizioni prodotti)
            api_key: Chiave API OpenAI
        """
        self.df_predictions = df_predictions
        self.client = OpenAI(api_key=api_key)
        self.model = OPENAI_MODEL_RESPONSE

        # Inizializza PyTrends
        if PYTRENDS_AVAILABLE:
            self.pytrends = TrendReq(hl='it-IT', tz=60)  # Italia timezone
        else:
            self.pytrends = None

        # Cache descrizioni prodotti
        self.product_descriptions = self._load_product_descriptions()

    def _load_product_descriptions(self) -> Dict[str, str]:
        """Carica le descrizioni dei prodotti dal CSV"""
        descriptions = {}

        if self.df_predictions is not None and 'Prodotto' in self.df_predictions.columns:
            if 'Descrizione_Prodotto' in self.df_predictions.columns:
                # Raggruppa per prodotto e prendi la prima descrizione
                prod_desc = self.df_predictions[['Prodotto', 'Descrizione_Prodotto']].drop_duplicates('Prodotto')
                descriptions = dict(zip(
                    prod_desc['Prodotto'].astype(str),
                    prod_desc['Descrizione_Prodotto']
                ))
                print(f"✅ Caricate {len(descriptions)} descrizioni prodotti")
            else:
                print("⚠️  Colonna Descrizione_Prodotto non trovata nel CSV")

        return descriptions

    def analyze_products_trends(
        self,
        prodotti: List[str],
        periodo: Dict[str, int]
    ) -> Dict:
        """
        Analizza i trends per una lista di prodotti

        Args:
            prodotti: Lista di codici prodotto
            periodo: Dict con 'mese' e 'anno'

        Returns:
            Dict con keywords e dati trends per ogni prodotto
        """
        results = {}

        for prodotto in prodotti:
            print(f"\n🔍 Analizzando trends per prodotto {prodotto}...")

            # 1. Ottieni descrizione prodotto
            descrizione = self.product_descriptions.get(str(prodotto), f"Prodotto {prodotto}")

            # 2. Genera keywords con GPT
            keywords = self._generate_keywords(prodotto, descrizione)

            # 3. Ottieni dati Google Trends
            trends_data = None
            if self.pytrends and keywords:
                trends_data = self._get_trends_data(keywords, periodo)

            results[prodotto] = {
                'descrizione': descrizione,
                'keywords': keywords,
                'trends_data': trends_data
            }

        return results

    def _generate_keywords(
        self,
        prodotto: str,
        descrizione: str
    ) -> List[str]:
        """
        Usa GPT per generare 5 keywords per Google Trends

        Args:
            prodotto: Codice prodotto
            descrizione: Descrizione del prodotto

        Returns:
            Lista di 5 keywords
        """
        try:
            system_prompt = """Sei un esperto di marketing per Gentilini, azienda italiana di biscotti premium.

Il tuo compito è generare 5 parole chiave SPECIFICHE da cercare su Google Trends per analizzare
fattori esogeni che potrebbero influenzare le vendite di un prodotto.

REGOLE CRITICHE:
1. Le keywords DEVONO essere STRETTAMENTE legate agli ingredienti/caratteristiche SPECIFICHE del prodotto
2. Analizza attentamente la descrizione del prodotto e identifica gli elementi chiave (es: cioccolato, cereali, integrali, frollini)
3. Le keywords devono riflettere ciò che i consumatori cercano quando vogliono QUEL TIPO SPECIFICO di biscotto
4. NON usare keywords generiche come "biscotti italiani", "merenda sana" a meno che siano direttamente correlate
5. Priorità agli ingredienti distintivi e al tipo di prodotto

ESEMPI:
- Prodotto: "Osvego Cioccolato" → ["biscotti cioccolato", "cookies cioccolato fondente", "biscotti cacao", "snack cioccolato", "dolci cioccolato"]
- Prodotto: "Frollini Burro" → ["frollini burro", "biscotti burro", "frollini classici", "biscotti tradizionali", "pasticcini burro"]
- Prodotto: "Cereali Croccanti" → ["biscotti cereali", "cookies cereali", "biscotti avena", "cereali colazione", "snack cereali"]
- Prodotto: "Integrali" → ["biscotti integrali", "frollini integrali", "biscotti fibra", "snack integrali", "cereali integrali"]

Le keywords devono essere:
- In italiano
- Termini che le persone cercano realmente su Google
- Max 3-4 parole per keyword
- Focalizzate sul TIPO SPECIFICO di prodotto

Rispondi SOLO con un array JSON di 5 stringhe, nient'altro.
"""

            user_prompt = f"""Prodotto: {prodotto}
Descrizione completa: {descrizione}

Analizza attentamente la descrizione e genera 5 keywords SPECIFICHE per questo prodotto.
Concentrati sugli ingredienti distintivi e il tipo di biscotto.

Keywords per Google Trends:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,  # Ridotto per più consistenza
                max_tokens=200
            )

            keywords_text = response.choices[0].message.content.strip()

            # Parse JSON
            keywords = json.loads(keywords_text)

            if isinstance(keywords, list) and len(keywords) == 5:
                print(f"✅ Keywords generate: {keywords}")
                return keywords
            else:
                print(f"⚠️  Formato keywords non valido, uso default")
                return self._get_default_keywords(descrizione)

        except Exception as e:
            print(f"❌ Errore generazione keywords: {e}")
            return self._get_default_keywords(descrizione)

    def _get_default_keywords(self, descrizione: str) -> List[str]:
        """Keywords di default basate sulla descrizione - ora più specifiche"""
        desc_lower = descrizione.lower()

        # Identifica ingredienti/caratteristiche chiave
        keywords = []

        # Cereali
        if any(word in desc_lower for word in ['cereali', 'cereal', 'avena', 'muesli']):
            keywords.extend(['biscotti cereali', 'cookies cereali', 'biscotti avena'])

        # Cioccolato
        elif any(word in desc_lower for word in ['cioccolato', 'cacao', 'chocolate']):
            keywords.extend(['biscotti cioccolato', 'cookies cioccolato', 'biscotti cacao'])

        # Integrali/Fibra
        elif any(word in desc_lower for word in ['integrale', 'integrali', 'fibra']):
            keywords.extend(['biscotti integrali', 'frollini integrali', 'biscotti fibra'])

        # Frollini
        elif 'frollin' in desc_lower:
            keywords.extend(['frollini burro', 'biscotti burro', 'frollini classici'])

        # Wafer
        elif 'wafer' in desc_lower:
            keywords.extend(['wafer', 'biscotti wafer', 'wafer cioccolato'])

        # Default generico (solo se nessuna corrispondenza)
        else:
            keywords.extend(['biscotti italiani', 'biscotti artigianali'])

        # Aggiungi keywords complementari
        keywords.extend(['colazione', 'merenda'])

        # Prendi solo le prime 5
        return keywords[:5]

    def _get_trends_data(
        self,
        keywords: List[str],
        periodo: Dict[str, int]
    ) -> Optional[Dict]:
        """
        Ottiene dati da Google Trends

        Args:
            keywords: Lista di keywords
            periodo: Dict con 'mese' e 'anno'

        Returns:
            Dict con dati trends o None se errore
        """
        if not self.pytrends:
            print("⚠️  PyTrends non disponibile")
            return None

        try:
            # Calcola timeframe
            anno = periodo['anno']
            mese = periodo['mese']

            # Data target
            target_date = datetime(anno, mese, 1)
            oggi = datetime.now()

            # Se il periodo è nel futuro, usa l'ultimo periodo disponibile
            if target_date > oggi:
                print(f"⚠️  Periodo {mese}/{anno} nel futuro, uso dati recenti")
                target_date = oggi
                # Vai indietro di 1 mese per avere dati completi
                if target_date.month == 1:
                    target_date = target_date.replace(year=target_date.year-1, month=12)
                else:
                    target_date = target_date.replace(month=target_date.month-1)

            # 3 mesi prima del target
            start_date = target_date - timedelta(days=90)

            # Formato per pytrends: 'YYYY-MM-DD YYYY-MM-DD'
            timeframe = f"{start_date.strftime('%Y-%m-%d')} {target_date.strftime('%Y-%m-%d')}"

            print(f"📊 Recupero trends per periodo: {timeframe}")

            # Build payload (max 5 keywords per query)
            self.pytrends.build_payload(
                keywords,
                cat=0,
                timeframe=timeframe,
                geo='IT',  # Italia
                gprop=''
            )

            # Interest over time
            interest_over_time = self.pytrends.interest_over_time()

            if interest_over_time.empty:
                print("⚠️  Nessun dato disponibile per le keywords")
                return None

            # Calcola medie per ogni keyword
            trends_summary = {}
            for keyword in keywords:
                if keyword in interest_over_time.columns:
                    values = interest_over_time[keyword].values
                    trends_summary[keyword] = {
                        'media': float(values.mean()),
                        'max': float(values.max()),
                        'min': float(values.min()),
                        'trend': 'crescente' if values[-1] > values[0] else 'decrescente',
                        'variazione': float(((values[-1] - values[0]) / (values[0] + 1)) * 100)
                    }

            print(f"✅ Dati trends recuperati per {len(trends_summary)} keywords")

            return {
                'timeframe': timeframe,
                'keywords_data': trends_summary,
                'periodo_target': f"{mese}/{anno}",
                'periodo_analisi_reale': f"{target_date.month}/{target_date.year}",
                'is_future': target_date > oggi
            }

        except Exception as e:
            print(f"❌ Errore recupero dati Google Trends: {e}")
            return None

    def format_trends_for_llm(self, trends_results: Dict) -> str:
        """
        Formatta i risultati trends per il prompt GPT

        Args:
            trends_results: Risultati dall'analisi trends

        Returns:
            Stringa formattata per il prompt
        """
        if not trends_results:
            return "Nessun dato Google Trends disponibile."

        output_parts = ["📊 FATTORI ESOGENI (Google Trends):\n"]

        for prodotto, data in trends_results.items():
            output_parts.append(f"\n🔸 Prodotto {prodotto}: {data['descrizione']}")

            if data['keywords']:
                output_parts.append(f"   Keywords analizzate: {', '.join(data['keywords'])}")

            if data['trends_data'] and data['trends_data'].get('keywords_data'):
                output_parts.append(f"   Periodo analisi: {data['trends_data']['timeframe']}")

                for keyword, metrics in data['trends_data']['keywords_data'].items():
                    trend_emoji = "📈" if metrics['trend'] == 'crescente' else "📉"
                    output_parts.append(
                        f"   {trend_emoji} '{keyword}': interesse medio {metrics['media']:.0f}/100, "
                        f"trend {metrics['trend']} ({metrics['variazione']:+.1f}%)"
                    )

        return "\n".join(output_parts)


if __name__ == "__main__":
    # Test del modulo
    from dotenv import load_dotenv
    load_dotenv()

    # Carica CSV per test
    import os
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        analyzer = TrendsAnalyzer(df)

        # Test con un prodotto
        test_prodotti = ['40000']
        test_periodo = {'mese': 1, 'anno': 2024}

        print("\n=== TEST TRENDS ANALYZER ===\n")
        results = analyzer.analyze_products_trends(test_prodotti, test_periodo)

        print("\n=== RISULTATI ===")
        formatted = analyzer.format_trends_for_llm(results)
        print(formatted)
    else:
        print(f"❌ CSV non trovato: {csv_path}")
