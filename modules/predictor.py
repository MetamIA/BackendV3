"""
Sistema Predittivo ML
Gestisce il caricamento del modello e la generazione di nuove predizioni
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import MODEL_PATH, PREDICTIONS_CSV


class MLPredictor:
    """
    Classe per gestire le predizioni usando il modello ML
    """
    
    def __init__(self, model_path: str = str(MODEL_PATH)):
        """
        Inizializza il predictor caricando il modello
        
        Args:
            model_path: Path al file del modello .pkl
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.performance = {}
        
        self._load_model()
    
    def _load_model(self):
        """Carica il modello e i suoi componenti"""
        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️ Modello non trovato in: {self.model_path}")
                print("Le predizioni non saranno disponibili")
                return
            
            model_package = joblib.load(self.model_path)
            
            self.model = model_package.get('model')
            self.scaler = model_package.get('scaler')
            self.label_encoders = model_package.get('label_encoders', {})
            self.feature_columns = model_package.get('features', 
                                                     model_package.get('feature_columns', []))
            self.performance = model_package.get('performance', {})
            
            model_name = model_package.get('model_name', 'GradientBoosting')
            print(f"✅ Modello {model_name} caricato")
            if self.performance:
                print(f"   R²: {self.performance.get('r2', 0):.4f}")
                print(f"   MAE: {self.performance.get('mae', 0):.2f} kg")
            print(f"   Features: {len(self.feature_columns)}")
            
        except Exception as e:
            print(f"❌ Errore caricamento modello: {e}")
            self.model = None
    
    def is_ready(self) -> bool:
        """Verifica se il predictor è pronto"""
        return self.model is not None
    
    def predict(
        self, 
        prodotto: str,
        cliente: str, 
        anno: int,
        mese: int,
        df_storico: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Genera una predizione per la combinazione prodotto-cliente-periodo
        
        Args:
            prodotto: Codice o nome del prodotto
            cliente: Codice o nome del cliente
            anno: Anno della predizione
            mese: Mese della predizione (1-12)
            df_storico: DataFrame con dati storici (opzionale)
            
        Returns:
            Tuple (risultato_dict, errore_string)
        """
        if not self.is_ready():
            return None, "Modello ML non disponibile"
        
        try:
            # Prepara i dati per la predizione
            input_data = self._prepare_input_data(
                prodotto, cliente, anno, mese, df_storico
            )
            
            if input_data is None:
                return None, "Impossibile preparare i dati per la predizione"
            
            # Esegui la predizione
            kg_predetti = self.model.predict(input_data)[0]
            
            # Costruisci il risultato
            risultato = {
                "prodotto": prodotto,
                "cliente": cliente,
                "anno": anno,
                "mese": mese,
                "kg_predetti": float(kg_predetti),
                "data_predizione": datetime.now().isoformat(),
                "modello": "GradientBoosting",
                "confidenza": self._calculate_confidence(kg_predetti)
            }
            
            return risultato, None
            
        except Exception as e:
            return None, f"Errore durante la predizione: {str(e)}"
    
    def _prepare_input_data(
        self, 
        prodotto: str,
        cliente: str,
        anno: int,
        mese: int,
        df_storico: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """
        Prepara i dati di input per il modello
        
        Questa è una versione semplificata. In produzione, 
        dovresti calcolare tutte le features del modello (lag, medie mobili, etc.)
        """
        try:
            # Crea un DataFrame con le feature base
            input_dict = {
                'Prodotto': [prodotto],
                'Cliente': [cliente],
                'Periodo': [mese],
                'Esercizio': [anno],
            }
            
            # Aggiungi feature temporali base
            input_dict['Mese_Sin'] = [np.sin(2 * np.pi * mese / 12)]
            input_dict['Mese_Cos'] = [np.cos(2 * np.pi * mese / 12)]
            input_dict['Trimestre'] = [(mese - 1) // 3 + 1]
            
            # Se hai dati storici, calcola lag features
            if df_storico is not None:
                # Filtra dati storici per questo prodotto-cliente
                storico_subset = df_storico[
                    (df_storico['Prodotto'] == prodotto) & 
                    (df_storico['Cliente'] == cliente)
                ].sort_values('Periodo')
                
                if len(storico_subset) > 0:
                    # Calcola alcune lag features base
                    input_dict['Kg_Lag_1'] = [storico_subset['Kg_Venduti_Predetti'].iloc[-1] 
                                              if len(storico_subset) >= 1 else 0]
                    input_dict['Kg_Lag_3'] = [storico_subset['Kg_Venduti_Predetti'].iloc[-3] 
                                              if len(storico_subset) >= 3 else 0]
                    input_dict['Media_Mobile_3'] = [storico_subset['Kg_Venduti_Predetti'].tail(3).mean() 
                                                   if len(storico_subset) >= 3 else 0]
            else:
                # Valori di default se non ci sono dati storici
                input_dict['Kg_Lag_1'] = [0]
                input_dict['Kg_Lag_3'] = [0]
                input_dict['Media_Mobile_3'] = [0]
            
            input_df = pd.DataFrame(input_dict)
            
            # Assicurati che ci siano tutte le feature del modello
            # Aggiungi colonne mancanti con valore 0
            for col in self.feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Riordina le colonne secondo l'ordine del modello
            input_df = input_df[self.feature_columns]
            
            # Applica scaling se disponibile
            if self.scaler is not None:
                input_df_scaled = self.scaler.transform(input_df)
                input_df = pd.DataFrame(input_df_scaled, columns=self.feature_columns)
            
            return input_df
            
        except Exception as e:
            print(f"Errore preparazione dati: {e}")
            return None
    
    def _calculate_confidence(self, prediction: float) -> float:
        """
        Calcola un punteggio di confidenza per la predizione
        
        Args:
            prediction: Valore predetto
            
        Returns:
            Confidenza tra 0 e 1
        """
        # Versione semplificata basata su MAE
        if not self.performance or 'mae' not in self.performance:
            return 0.7  # Default
        
        mae = self.performance['mae']
        
        # Calcola confidenza basata su quanto è lontana la predizione dall'errore medio
        # Più è grande la predizione rispetto all'errore, più siamo confidenti
        if prediction > 0:
            confidence = 1 - (mae / (prediction + mae))
            return max(0.5, min(1.0, confidence))
        
        return 0.5


if __name__ == "__main__":
    # Test del predictor
    predictor = MLPredictor()
    
    if predictor.is_ready():
        print("\n=== TEST DEL PREDICTOR ===\n")
        
        # Test con dati di esempio
        risultato, errore = predictor.predict(
            prodotto="40000",
            cliente="393",
            anno=2024,
            mese=1
        )
        
        if risultato:
            print("Predizione riuscita:")
            print(f"  Prodotto: {risultato['prodotto']}")
            print(f"  Cliente: {risultato['cliente']}")
            print(f"  Periodo: {risultato['mese']}/{risultato['anno']}")
            print(f"  Kg predetti: {risultato['kg_predetti']:.2f}")
            print(f"  Confidenza: {risultato['confidenza']:.2%}")
        else:
            print(f"Errore: {errore}")
    else:
        print("Predictor non disponibile")
