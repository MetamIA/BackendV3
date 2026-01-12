"""
Prediction Manager Enhanced - COMPLETAMENTE FIXATO
Fix: colonne CSV, periodo_range, aggregazioni, ML debug
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI
import json

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
    
    # Mapping colonne CSV Gentilini
    COL_PRODOTTO = 'Prodotto'
    COL_CLIENTE = 'Cliente'
    COL_PERIODO = 'Periodo'  # Mese (1-12)
    COL_ESERCIZIO = 'Esercizio'  # Anno
    COL_KG = 'Kg_Venduti_Predetti'
    
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
                
                # Debug: mostra colonne
                print(f"   Colonne CSV: {list(self.df_predictions.columns[:10])}")
            else:
                print(f"⚠️  CSV non trovato: {self.csv_path}")
                self.df_predictions = pd.DataFrame()
        except Exception as e:
            print(f"❌ Errore caricamento CSV: {e}")
            self.df_predictions = pd.DataFrame()
    
    def execute_operation(self, operation: str, input_data: Dict) -> Dict:
        """
        Esegue operazione richiesta dall'orchestrator
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
        Predizione per un singolo prodotto-cliente-periodo
        """
        prodotti = input_data.get('prodotti', [])
        clienti = input_data.get('clienti', [])
        periodo = input_data.get('periodo', {})
        
        # Espandi prodotti
        prodotti_expanded = self._expand_products(prodotti)
        
        # Gestisci clienti="*"
        if clienti == '*':
            return self._aggregazione_clienti(input_data)
        
        # Espandi clienti
        clienti_expanded = self._expand_clients(clienti)
        
        risultati = []
        
        for prodotto in prodotti_expanded[:5]:
            for cliente in clienti_expanded[:10]:
                pred, fonte = self._get_or_create_prediction(
                    str(prodotto),
                    str(cliente),
                    periodo['anno'],
                    periodo['mese']
                )
                
                if pred:
                    confidenza = pred.get('confidenza') if fonte == 'ml_model' else None
                    
                    risultati.append({
                        'prodotto': str(prodotto),
                        'cliente': str(cliente),
                        'periodo': periodo,
                        'kg_predetti': float(pred['kg_predetti']),
                        'confidenza': confidenza,
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
        periodo = input_data.get('periodo')
        periodo_range = input_data.get('periodo_range')
        aggregazione = input_data.get('aggregazione', 'somma')
        
        print(f"   Aggregazione '{aggregazione}' su tutti i clienti")
        
        # Se periodo_range → converti in operazione somma/media sul range
        if periodo_range and not periodo:
            print(f"   → Ricevuto periodo_range, converto in operazione range")
            input_data_converted = input_data.copy()
            input_data_converted['clienti'] = '*'
            
            if aggregazione == 'somma':
                return self._operazione_range(input_data_converted, "somma")
            elif aggregazione == 'media':
                return self._operazione_range(input_data_converted, "media")
            else:
                return self._operazione_range(input_data_converted, "mediana")
        
        # Altrimenti periodo singolo
        if not periodo:
            return {'operazione': 'aggregazione_clienti', 'risultati': [], 'num_risultati': 0}
        
        prodotti_expanded = self._expand_products(prodotti)
        
        risultati = []
        
        for prodotto in prodotti_expanded[:5]:
            # Cerca tutti i clienti per questo prodotto-periodo nel CSV
            if self.df_predictions is not None and not self.df_predictions.empty:
                mask = (
                    (self.df_predictions[self.COL_PRODOTTO].astype(str) == str(prodotto)) &
                    (self.df_predictions[self.COL_PERIODO] == periodo['mese']) &
                    (self.df_predictions[self.COL_ESERCIZIO] == periodo['anno'])
                )
                
                df_match = self.df_predictions[mask]
                
                if not df_match.empty:
                    kg_values = df_match[self.COL_KG].values
                    
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
                            'confidenza': None
                        })
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
        
        # Gestisci periodo_range o periodo con struttura {da, a}
        periodo_range = input_data.get('periodo_range')
        if not periodo_range:
            periodo = input_data.get('periodo', {})
            if 'da' in periodo and 'a' in periodo:
                periodo_range = periodo
            else:
                print(f"   ❌ Nessun range trovato")
                return {'operazione': operazione, 'risultati': [], 'num_risultati': 0}
        
        print(f"   Range: {periodo_range['da']} → {periodo_range['a']}")
        
        prodotti_expanded = self._expand_products(prodotti)
        
        # Gestisci clienti
        if clienti == '*':
            clienti_list = ['*']
        else:
            clienti_list = self._expand_clients(clienti)
        
        risultati = []
        
        # Genera lista periodi nel range
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
                        # Aggregazione tutti i clienti per questo periodo
                        if self.df_predictions is not None and not self.df_predictions.empty:
                            mask = (
                                (self.df_predictions[self.COL_PRODOTTO].astype(str) == str(prodotto)) &
                                (self.df_predictions[self.COL_PERIODO] == periodo['mese']) &
                                (self.df_predictions[self.COL_ESERCIZIO] == periodo['anno'])
                            )
                            
                            df_match = self.df_predictions[mask]
                            
                            if not df_match.empty:
                                kg_periodo = df_match[self.COL_KG].sum()
                                if pd.notna(kg_periodo):
                                    kg_values.append(float(kg_periodo))
                                    fonti.append('csv')
                    else:
                        # Cliente specifico
                        pred, fonte = self._get_or_create_prediction(
                            str(prodotto),
                            str(cliente),
                            periodo['anno'],
                            periodo['mese']
                        )
                        
                        if pred:
                            kg_values.append(float(pred['kg_predetti']))
                            fonti.append(fonte)
                
                if len(kg_values) > 0:
                    # Calcola valore aggregato
                    if operazione == 'somma':
                        valore = float(np.sum(kg_values))
                    elif operazione == 'media':
                        valore = float(np.mean(kg_values))
                    else:
                        valore = float(np.median(kg_values))
                    
                    # Determina tipo dato
                    has_ml = 'ml_model' in fonti
                    tipo_dato = 'predizione' if has_ml else 'storico'
                    
                    risultato = {
                        'prodotto': str(prodotto),
                        'valore': valore,
                        'num_periodi': len(kg_values),
                        'operazione': operazione,
                        'tipo_dato': tipo_dato,
                        'confidenza': None,
                        'periodo_range': periodo_range
                    }
                    
                    if cliente == '*':
                        risultato['clienti'] = 'tutti'
                        # Conta clienti unici
                        if self.df_predictions is not None:
                            mask_all = (
                                (self.df_predictions[self.COL_PRODOTTO].astype(str) == str(prodotto))
                            )
                            num_clienti = self.df_predictions[mask_all][self.COL_CLIENTE].nunique()
                            risultato['num_clienti'] = num_clienti
                    else:
                        risultato['cliente'] = str(cliente)
                    
                    risultati.append(risultato)
                else:
                    print(f"   ⚠️  Nessun dato per prodotto {prodotto}, cliente {cliente}")
        
        return {
            'operazione': operazione,
            'risultati': risultati,
            'num_risultati': len(risultati)
        }
    
    def _generate_period_range(self, da: Dict, a: Dict) -> List[Dict]:
        """
        Genera lista di periodi tra da e a
        """
        periodi = []
        
        anno_start = da['anno']
        mese_start = da['mese']
        anno_end = a['anno']
        mese_end = a['mese']
        
        anno_current = anno_start
        mese_current = mese_start
        
        while (anno_current < anno_end) or (anno_current == anno_end and mese_current <= mese_end):
            periodi.append({
                'anno': anno_current,
                'mese': mese_current
            })
            
            mese_current += 1
            if mese_current > 12:
                mese_current = 1
                anno_current += 1
        
        return periodi
    
    def _expand_products(self, prodotti: List) -> List:
        """Espandi lista prodotti"""
        if not prodotti or prodotti == ['*']:
            return ['*']
        
        expanded = []
        for prod in prodotti:
            if prod == '*':
                expanded.append('*')
            else:
                expanded.append(str(prod))
        
        return expanded
    
    def _expand_clients(self, clienti: List) -> List:
        """Espandi lista clienti"""
        if not clienti or clienti == ['*']:
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
                (self.df_predictions[self.COL_PRODOTTO].astype(str) == str(prodotto)) &
                (self.df_predictions[self.COL_CLIENTE].astype(str) == str(cliente)) &
                (self.df_predictions[self.COL_PERIODO] == mese) &
                (self.df_predictions[self.COL_ESERCIZIO] == anno)
            )
            
            matches = self.df_predictions[mask]
            
            if not matches.empty:
                row = matches.iloc[0]
                kg = row.get(self.COL_KG)
                
                if pd.notna(kg):
                    return {
                        'kg_predetti': float(kg),
                    }, 'csv'
        
        # Genera con ML
        try:
            print(f"   ⚠️  Non in CSV, chiamo ML: {prodotto}/{cliente}/{anno}/{mese}")
            prediction = self.predictor.predict(prodotto, cliente, anno, mese)
            
            print(f"   🔍 ML ha restituito: {prediction}")
            print(f"   🔍 Tipo: {type(prediction)}")
            
            # Se è una tupla, prendi il primo elemento
            if isinstance(prediction, tuple):
                prediction = prediction[0]
                print(f"   → Estraggo primo elemento tupla: {prediction}")
            
            if prediction:
                # Prova vari formati possibili
                kg_value = None
                
                # Formato 1: dict con 'kg_predetti'
                if isinstance(prediction, dict) and 'kg_predetti' in prediction:
                    kg_value = prediction['kg_predetti']
                    confidenza = prediction.get('confidenza', 0.75)
                
                # Formato 2: dict con 'kg'
                elif isinstance(prediction, dict) and 'kg' in prediction:
                    kg_value = prediction['kg']
                    confidenza = prediction.get('confidenza', 0.75)
                
                # Formato 3: numero diretto
                elif isinstance(prediction, (int, float)):
                    kg_value = prediction
                    confidenza = 0.75
                
                # Formato 4: numpy array
                elif hasattr(prediction, 'shape'):
                    kg_value = float(prediction.flatten()[0])
                    confidenza = 0.75
                
                if kg_value is not None:
                    print(f"   ✓ ML predetto: {kg_value:.2f} kg (confidenza: {confidenza:.0%})")
                    return {
                        'kg_predetti': float(kg_value),
                        'confidenza': confidenza
                    }, 'ml_model'
                else:
                    print(f"   ✗ Prediction non ha formato riconosciuto")
                    print(f"   ✗ Keys: {prediction.keys() if isinstance(prediction, dict) else 'N/A'}")
            else:
                print(f"   ✗ ML ha restituito None")
        except Exception as e:
            print(f"   ✗ Errore ML: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"   ✗ Nessun dato per {prodotto}/{cliente}/{anno}/{mese}")
        return None, None


if __name__ == "__main__":
    # Test
    from dotenv import load_dotenv
    load_dotenv()
    
    manager = PredictionManagerEnhanced()
    
    # Test
    print("\n=== TEST: Aggregazione clienti ===")
    result = manager.execute_operation('aggregazione_clienti', {
        'prodotti': ['40003'],
        'periodo': {'mese': 1, 'anno': 2024},
        'aggregazione': 'somma'
    })
    print(json.dumps(result, indent=2))
