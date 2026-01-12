"""
Prediction Manager Enhanced - Aggregazioni e Operazioni Intelligenti
Gestisce: clienti="*", somme, medie, mediane, range periodi
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    OPENAI_API_KEY, 
    OPENAI_MODEL_RESPONSE,
    PREDICTIONS_CSV
)
from modules.predictor import MLPredictor


class PredictionManagerEnhanced:
    """
    Manager con supporto aggregazioni e operazioni avanzate
    """
    
    def __init__(
        self, 
        csv_path: str = str(PREDICTIONS_CSV),
        api_key: str = OPENAI_API_KEY
    ):
        """Inizializza manager"""
        self.csv_path = csv_path
        self.df_predictions = None
        self.predictor = MLPredictor()
        self.client = OpenAI(api_key=api_key)
        self.model = OPENAI_MODEL_RESPONSE
        
        # Carica CSV
        self._load_predictions_csv()
    
    def _load_predictions_csv(self):
        """Carica CSV predizioni"""
        try:
            if os.path.exists(self.csv_path):
                self.df_predictions = pd.read_csv(self.csv_path)
                print(f"✅ CSV predizioni caricato: {len(self.df_predictions)} righe")
            else:
                print(f"⚠️  CSV non trovato: {self.csv_path}")
                self.df_predictions = pd.DataFrame()
        except Exception as e:
            print(f"❌ Errore caricamento CSV: {e}")
            self.df_predictions = pd.DataFrame()
    
    def execute_operation(self, operation: str, input_data: Dict) -> Dict:
        """
        Esegue operazione richiesta dall'orchestrator
        
        Args:
            operation: Tipo operazione
            input_data: Input preparato da LLM
            
        Returns:
            Dict con risultati
        """
        
        print(f"\n📊 Esecuzione operazione: {operation}")
        
        if operation == "predizione_singola":
            return self._predizione_singola(input_data)
        
        elif operation == "aggregazione_clienti":
            return self._aggregazione_clienti(input_data)
        
        elif operation == "somma":
            return self._operazione_range(input_data, "somma")
        
        elif operation == "media":
            return self._operazione_range(input_data, "media")
        
        elif operation == "mediana":
            return self._operazione_range(input_data, "mediana")
        
        else:
            return {'error': f'Operazione non supportata: {operation}'}
    
    def _predizione_singola(self, input_data: Dict) -> Dict:
        """
        Predizione per singolo prodotto-cliente-periodo
        """
        prodotti = input_data.get('prodotti', [])
        clienti = input_data.get('clienti', [])
        periodo = input_data.get('periodo', {})
        
        # Espandi prodotti se necessario
        prodotti_expanded = self._expand_products(prodotti)
        
        # Gestisci clienti="*"
        if clienti == '*':
            return self._aggregazione_clienti(input_data)
        
        # Espandi clienti
        clienti_expanded = self._expand_clients(clienti)
        
        risultati = []
        
        for prodotto in prodotti_expanded[:5]:  # Max 5
            for cliente in clienti_expanded[:10]:  # Max 10
                pred, fonte = self._get_or_create_prediction(
                    str(prodotto),
                    str(cliente),
                    periodo['anno'],
                    periodo['mese']
                )
                
                if pred:
                    # Confidenza solo se ML, non se CSV
                    confidenza = pred.get('confidenza') if fonte == 'ml_model' else None
                    
                    risultati.append({
                        'prodotto': str(prodotto),
                        'cliente': str(cliente),
                        'periodo': periodo,
                        'kg_predetti': float(pred['kg_predetti']),
                        'confidenza': confidenza,  # None per dati storici
                        'fonte': fonte,
                        'tipo_dato': 'storico' if fonte == 'csv' else 'predizione'
                    })
        
        return {
            'operazione': 'predizione_singola',
            'risultati': risultati,
            'num_risultati': len(risultati)
        }
    
    def _aggregazione_clienti(self, input_data: Dict) -> Dict:
        """
        Aggregazione su tutti i clienti (somma o media)
        """
        prodotti = input_data.get('prodotti', [])
        periodo = input_data.get('periodo', {})
        aggregazione = input_data.get('aggregazione', 'somma')  # Default somma
        
        print(f"   Aggregazione '{aggregazione}' su tutti i clienti")
        
        prodotti_expanded = self._expand_products(prodotti)
        
        risultati = []
        
        for prodotto in prodotti_expanded[:5]:
            # Cerca tutti i clienti per questo prodotto-periodo nel CSV
            if self.df_predictions is not None and not self.df_predictions.empty:
                mask = (
                    (self.df_predictions['Prodotto'].astype(str) == str(prodotto)) &
                    (self.df_predictions['Mese'] == periodo['mese']) &
                    (self.df_predictions['Anno'] == periodo['anno'])
                )
                
                df_match = self.df_predictions[mask]
                
                if not df_match.empty:
                    kg_values = df_match['Kg_Venduti_Predetti'].values
                    
                    # Rimuovi NaN
                    kg_values = kg_values[~pd.isna(kg_values)]
                    
                    if len(kg_values) > 0:
                        if aggregazione == 'somma':
                            valore = float(np.sum(kg_values))
                        elif aggregazione == 'media':
                            valore = float(np.mean(kg_values))
                        else:
                            valore = float(np.median(kg_values))
                        
                        risultati.append({
                            'prodotto': str(prodotto),
                            'clienti': 'tutti',
                            'num_clienti': len(kg_values),
                            'periodo': periodo,
                            'valore': valore,
                            'operazione': aggregazione,
                            'fonte': 'csv',
                            'tipo_dato': 'storico',
                            'confidenza': None  # Dati storici = certi
                        })
                    else:
                        print(f"   ⚠️  Nessun dato valido per prodotto {prodotto}")
                else:
                    print(f"   ⚠️  Nessun match nel CSV per prodotto {prodotto}")
            else:
                print(f"   ⚠️  CSV non disponibile")
        
        return {
            'operazione': f'aggregazione_clienti_{aggregazione}',
            'risultati': risultati,
            'num_risultati': len(risultati)
        }
    
    def _operazione_range(self, input_data: Dict, operazione: str) -> Dict:
        """
        Operazione su range di periodi (somma, media, mediana)
        """
        prodotti = input_data.get('prodotti', [])
        clienti = input_data.get('clienti', [])
        periodo_range = input_data.get('periodo_range', {})
        
        print(f"   Range: {periodo_range['da']} → {periodo_range['a']}")
        
        prodotti_expanded = self._expand_products(prodotti)
        
        # Gestisci clienti
        if clienti == '*':
            clienti_list = ['*']  # Aggregazione tutti
        else:
            clienti_list = self._expand_clients(clienti)
        
        risultati = []
        
        # Genera lista di periodi nel range
        periodi_list = self._generate_period_range(
            periodo_range['da'],
            periodo_range['a']
        )
        
        print(f"   Periodi da analizzare: {len(periodi_list)}")
        
        for prodotto in prodotti_expanded[:5]:
            for cliente in clienti_list[:10]:
                # Recupera dati per tutti i periodi
                kg_values = []
                fonti = []
                
                for periodo in periodi_list:
                    if cliente == '*':
                        # Aggregazione su tutti i clienti per questo periodo
                        kg_agg = self._get_aggregated_for_period(
                            prodotto, periodo, 'somma'
                        )
                        if kg_agg is not None:
                            kg_values.append(kg_agg)
                            fonti.append('csv')
                    else:
                        pred, fonte = self._get_or_create_prediction(
                            str(prodotto),
                            str(cliente),
                            periodo['anno'],
                            periodo['mese']
                        )
                        
                        if pred and pred.get('kg_predetti'):
                            kg_values.append(float(pred['kg_predetti']))
                            fonti.append(fonte)
                
                if kg_values:
                    # Calcola operazione
                    if operazione == 'somma':
                        valore_finale = float(np.sum(kg_values))
                    elif operazione == 'media':
                        valore_finale = float(np.mean(kg_values))
                    elif operazione == 'mediana':
                        valore_finale = float(np.median(kg_values))
                    else:
                        valore_finale = float(np.sum(kg_values))
                    
                    # Confidenza: se tutti CSV → None, altrimenti media
                    if all(f == 'csv' for f in fonti):
                        confidenza = None
                        tipo_dato = 'storico'
                    else:
                        confidenza = 0.75  # Confidenza media per mix dati
                        tipo_dato = 'misto'
                    
                    risultati.append({
                        'prodotto': str(prodotto),
                        'cliente': 'tutti' if cliente == '*' else str(cliente),
                        'periodo_range': periodo_range,
                        'num_periodi': len(kg_values),
                        'valore': valore_finale,
                        'operazione': operazione,
                        'tipo_dato': tipo_dato,
                        'confidenza': confidenza,
                        'dettaglio_periodi': kg_values if len(kg_values) <= 12 else None
                    })
        
        return {
            'operazione': operazione,
            'risultati': risultati,
            'num_risultati': len(risultati)
        }
    
    def _get_aggregated_for_period(
        self, 
        prodotto: str, 
        periodo: Dict,
        aggregazione: str = 'somma'
    ) -> Optional[float]:
        """Ottiene valore aggregato per tutti i clienti in un periodo"""
        if self.df_predictions is None or self.df_predictions.empty:
            return None
        
        mask = (
            (self.df_predictions['Prodotto'].astype(str) == str(prodotto)) &
            (self.df_predictions['Mese'] == periodo['mese']) &
            (self.df_predictions['Anno'] == periodo['anno'])
        )
        
        df_match = self.df_predictions[mask]
        
        if df_match.empty:
            return None
        
        kg_values = df_match['Kg_Venduti_Predetti'].values
        kg_values = kg_values[~pd.isna(kg_values)]
        
        if len(kg_values) == 0:
            return None
        
        if aggregazione == 'somma':
            return float(np.sum(kg_values))
        elif aggregazione == 'media':
            return float(np.mean(kg_values))
        else:
            return float(np.median(kg_values))
    
    def _generate_period_range(self, da: Dict, a: Dict) -> List[Dict]:
        """Genera lista di periodi da 'da' ad 'a'"""
        periodi = []
        
        anno_start = da['anno']
        mese_start = da['mese']
        anno_end = a['anno']
        mese_end = a['mese']
        
        anno = anno_start
        mese = mese_start
        
        while (anno < anno_end) or (anno == anno_end and mese <= mese_end):
            periodi.append({'mese': mese, 'anno': anno})
            
            mese += 1
            if mese > 12:
                mese = 1
                anno += 1
            
            # Safety: max 60 mesi (5 anni)
            if len(periodi) > 60:
                break
        
        return periodi
    
    def _expand_products(self, prodotti: List) -> List[str]:
        """Espande ricerche testuali in codici prodotto"""
        expanded = []
        
        for prod in prodotti:
            if prod == '*':
                # Tutti i prodotti (max 10)
                if self.df_predictions is not None and not self.df_predictions.empty:
                    all_prods = self.df_predictions['Prodotto'].unique()
                    expanded.extend([str(p) for p in all_prods[:10]])
                continue
            
            if str(prod).isdigit():
                expanded.append(str(prod))
            else:
                # Cerca per descrizione
                if self.df_predictions is not None and not self.df_predictions.empty:
                    if 'Descrizione_Prodotto' in self.df_predictions.columns:
                        matches = self.df_predictions[
                            self.df_predictions['Descrizione_Prodotto'].str.contains(
                                str(prod), case=False, na=False
                            )
                        ]['Prodotto'].unique()
                        
                        expanded.extend([str(p) for p in matches[:5]])
                
                if not expanded:
                    expanded.append(str(prod))
        
        return list(set(expanded))[:10]  # Max 10 prodotti unici
    
    def _expand_clients(self, clienti: List) -> List[str]:
        """Espande lista clienti"""
        if isinstance(clienti, str) and clienti == '*':
            return ['*']
        
        expanded = []
        for cliente in clienti:
            if cliente == '*':
                expanded.append('*')
            else:
                expanded.append(str(cliente))
        
        return expanded
    
    def _get_or_create_prediction(
        self,
        prodotto: str,
        cliente: str,
        anno: int,
        mese: int
    ) -> Tuple[Optional[Dict], str]:
        """
        Cerca nel CSV o genera con ML
        
        Returns:
            (predizione_dict, fonte)
            fonte: 'csv' o 'ml_model'
        """
        # Cerca nel CSV
        if self.df_predictions is not None and not self.df_predictions.empty:
            mask = (
                (self.df_predictions['Prodotto'].astype(str) == str(prodotto)) &
                (self.df_predictions['Cliente'].astype(str) == str(cliente)) &
                (self.df_predictions['Mese'] == mese) &
                (self.df_predictions['Anno'] == anno)
            )
            
            matches = self.df_predictions[mask]
            
            if not matches.empty:
                row = matches.iloc[0]
                kg = row.get('Kg_Venduti_Predetti')
                
                if pd.notna(kg):
                    return {
                        'kg_predetti': float(kg),
                        # NO confidenza per dati storici!
                    }, 'csv'
        
        # Genera con ML
        try:
            prediction = self.predictor.predict(prodotto, cliente, anno, mese)
            
            if prediction:
                return {
                    'kg_predetti': prediction['kg_predetti'],
                    'confidenza': prediction.get('confidenza', 0.75)
                }, 'ml_model'
        except:
            pass
        
        return None, None


if __name__ == "__main__":
    # Test
    from dotenv import load_dotenv
    load_dotenv()
    
    manager = PredictionManagerEnhanced()
    
    # Test 1: Cliente mancante
    print("\n=== TEST 1: Cliente mancante (aggregazione) ===")
    result = manager.execute_operation('aggregazione_clienti', {
        'prodotti': ['40000'],
        'periodo': {'mese': 1, 'anno': 2024},
        'aggregazione': 'somma'
    })
    print(json.dumps(result, indent=2))
    
    # Test 2: Somma range
    print("\n=== TEST 2: Somma Q1 2024 ===")
    result = manager.execute_operation('somma', {
        'prodotti': ['40000'],
        'clienti': ['10000018'],
        'periodo_range': {
            'da': {'mese': 1, 'anno': 2024},
            'a': {'mese': 3, 'anno': 2024}
        }
    })
    print(json.dumps(result, indent=2))
