"""
Name-Code Mapper - Partial Name Matching
Matcha nomi parziali come "Osvego 3500" -> "Osvego 3500 - 14pz x 250g"
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from difflib import SequenceMatcher
import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PREDICTIONS_CSV


class NameCodeMapper:
    """
    Mapper con matching parziale intelligente
    """
    
    def __init__(self, csv_path: str = str(PREDICTIONS_CSV)):
        """Inizializza mapper"""
        self.csv_path = csv_path
        
        self.product_code_to_name = {}
        self.client_code_to_name = {}
        self.product_name_to_code = {}
        self.client_name_to_code = {}
        self.product_names = []
        self.client_names = []
        self.product_names_lower = []
        self.client_names_lower = []
        
        self._load_mappings()
    
    def _load_mappings(self):
        """Carica mappings da CSV"""
        try:
            df = pd.read_csv(self.csv_path)
            
            # Prodotti
            product_mapping = df[['Prodotto', 'Descrizione_Prodotto']].drop_duplicates()
            for _, row in product_mapping.iterrows():
                code = str(row['Prodotto'])
                name = str(row['Descrizione_Prodotto'])
                
                if pd.notna(name) and name != 'nan':
                    self.product_code_to_name[code] = name
                    self.product_name_to_code[name.lower()] = code
                    self.product_names.append(name)
                    self.product_names_lower.append(name.lower())
            
            # Clienti
            client_mapping = df[['Cliente', 'Descrizione_Cliente']].drop_duplicates()
            for _, row in client_mapping.iterrows():
                code = str(row['Cliente'])
                name = str(row['Descrizione_Cliente'])
                
                if pd.notna(name) and name != 'nan':
                    self.client_code_to_name[code] = name
                    self.client_name_to_code[name.lower()] = code
                    self.client_names.append(name)
                    self.client_names_lower.append(name.lower())
            
            print(f"✅ NameCodeMapper inizializzato:")
            print(f"   - {len(self.product_code_to_name)} prodotti")
            print(f"   - {len(self.client_code_to_name)} clienti")
            
        except Exception as e:
            print(f"❌ Errore caricamento mappings: {e}")
    
    def translate_names_to_codes(self, text: str) -> Tuple[str, Dict]:
        """
        Traduce nomi in codici con matching parziale
        """
        translated_text = text
        metadata = {
            'prodotti_tradotti': [],
            'clienti_tradotti': []
        }
        
        used_product_codes: Set[str] = set()
        used_client_codes: Set[str] = set()
        
        # STEP 1: Match ESATTI prodotti
        for name in self.product_names:
            pattern = re.compile(re.escape(name), re.IGNORECASE)
            matches = list(pattern.finditer(translated_text))
            
            if matches:
                code = self.product_name_to_code.get(name.lower())
                if code and code not in used_product_codes:
                    translated_text = pattern.sub(f"prodotto {code}", translated_text)
                    used_product_codes.add(code)
                    
                    metadata['prodotti_tradotti'].append({
                        'nome': name,
                        'codice': code,
                        'tipo': 'esatto'
                    })
                    
                    print(f"   ✓ Prodotto match esatto: '{name}' → codice {code}")
        
        # STEP 2: Match PARZIALI prodotti (es. "Osvego 3500" matcha "Osvego 3500 - 14pz x 250g")
        if not metadata['prodotti_tradotti']:
            # Cerca match parziali solo se non trovato match esatto
            words_in_text = translated_text.split()
            
            # Cerca sequenze di 2-3 parole che potrebbero essere nomi prodotti
            for i in range(len(words_in_text)):
                for length in [3, 2, 1]:  # Prova 3 parole, poi 2, poi 1
                    if i + length > len(words_in_text):
                        continue
                    
                    potential_name = ' '.join(words_in_text[i:i+length])
                    
                    # Salta se troppo corto o contiene solo numeri
                    if len(potential_name) < 4:
                        continue
                    
                    # Cerca prodotti che contengono questo nome
                    for j, full_name_lower in enumerate(self.product_names_lower):
                        full_name = self.product_names[j]
                        
                        # Check se potential_name è contenuto all'inizio del nome completo
                        if full_name_lower.startswith(potential_name.lower()):
                            code = self.product_name_to_code.get(full_name_lower)
                            
                            if code and code not in used_product_codes:
                                # Sostituisci solo questa occorrenza
                                pattern = re.compile(r'\b' + re.escape(potential_name) + r'\b', re.IGNORECASE)
                                translated_text = pattern.sub(f"prodotto {code}", translated_text, count=1)
                                used_product_codes.add(code)
                                
                                metadata['prodotti_tradotti'].append({
                                    'nome': full_name,
                                    'codice': code,
                                    'tipo': 'parziale',
                                    'matched_text': potential_name
                                })
                                
                                print(f"   ✓ Prodotto match parziale: '{potential_name}' → '{full_name}' (codice {code})")
                                break
                        
                    if metadata['prodotti_tradotti']:
                        break
                
                if metadata['prodotti_tradotti']:
                    break
        
        # STEP 3: Match ESATTI clienti
        for name in self.client_names:
            pattern = re.compile(re.escape(name), re.IGNORECASE)
            matches = list(pattern.finditer(translated_text))
            
            if matches:
                code = self.client_name_to_code.get(name.lower())
                if code and code not in used_client_codes:
                    translated_text = pattern.sub(f"cliente {code}", translated_text)
                    used_client_codes.add(code)
                    
                    metadata['clienti_tradotti'].append({
                        'nome': name,
                        'codice': code,
                        'tipo': 'esatto'
                    })
        
        # STEP 4: Fuzzy match CONTESTUALE per clienti
        client_keywords = ['cliente', 'per', 'al', 'alla', 'del', 'della']
        
        words = translated_text.split()
        for i, word in enumerate(words):
            if 'prodotto' in word or 'cliente' in word or len(word) < 4:
                continue
            
            # Verifica contesto cliente
            is_client_context = False
            
            if i > 0:
                prev_words = ' '.join(words[max(0, i-3):i]).lower()
                if any(kw in prev_words for kw in client_keywords):
                    is_client_context = True
            
            if i > len(words) * 0.7:
                is_client_context = True
            
            if is_client_context:
                client_matches = self.fuzzy_match_client(word, threshold=0.75)
                
                if client_matches:
                    best_match = client_matches[0]
                    name, code, score = best_match
                    
                    if code not in used_client_codes:
                        print(f"   🔍 Fuzzy match cliente: '{word}' → '{name}' (score: {score:.2f})")
                        
                        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                        translated_text = pattern.sub(f"cliente {code}", translated_text, count=1)
                        used_client_codes.add(code)
                        
                        metadata['clienti_tradotti'].append({
                            'nome': name,
                            'codice': code,
                            'tipo': 'fuzzy',
                            'original_word': word,
                            'score': score
                        })
        
        return translated_text, metadata
    
    def fuzzy_match_product(self, query: str, threshold: float = 0.6) -> List[Tuple[str, str, float]]:
        """Trova prodotti simili"""
        matches = []
        query_lower = query.lower()
        
        for i, name_lower in enumerate(self.product_names_lower):
            name = self.product_names[i]
            score = SequenceMatcher(None, query_lower, name_lower).ratio()
            
            if query_lower in name_lower:
                score = max(score, 0.85)
            
            query_words = set(query_lower.split())
            name_words = set(name_lower.split())
            common_words = query_words & name_words
            if common_words:
                score = max(score, 0.7 + (len(common_words) * 0.05))
            
            if score >= threshold:
                code = self.product_name_to_code.get(name_lower)
                if code:
                    matches.append((name, code, score))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:10]
    
    def fuzzy_match_client(self, query: str, threshold: float = 0.6) -> List[Tuple[str, str, float]]:
        """Trova clienti simili"""
        matches = []
        query_lower = query.lower()
        
        for i, name_lower in enumerate(self.client_names_lower):
            name = self.client_names[i]
            score = SequenceMatcher(None, query_lower, name_lower).ratio()
            
            if query_lower in name_lower:
                score = max(score, 0.85)
            
            query_words = set(query_lower.split())
            name_words = set(name_lower.split())
            common_words = query_words & name_words
            if common_words:
                score = max(score, 0.7 + (len(common_words) * 0.05))
            
            if query_lower in [w.lower() for w in name.split()]:
                score = max(score, 0.9)
            
            if score >= threshold:
                code = self.client_name_to_code.get(name_lower)
                if code:
                    matches.append((name, code, score))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:10]
    
    def code_to_name_product(self, code: str) -> Optional[str]:
        """Converte codice prodotto in nome"""
        return self.product_code_to_name.get(str(code))
    
    def code_to_name_client(self, code: str) -> Optional[str]:
        """Converte codice cliente in nome"""
        return self.client_code_to_name.get(str(code))
    
    def translate_response(self, response_data: Dict) -> Dict:
        """Traduce codici in nomi nella risposta"""
        
        if 'risposta' in response_data and response_data['risposta']:
            text = response_data['risposta']
            
            for code, name in self.product_code_to_name.items():
                text = text.replace(f"prodotto {code}", name)
                text = text.replace(f"Prodotto {code}", name)
                text = re.sub(r'\b' + re.escape(code) + r'\b', name, text)
            
            for code, name in self.client_code_to_name.items():
                text = text.replace(f"cliente {code}", name)
                text = text.replace(f"Cliente {code}", name)
                text = re.sub(r'\b' + re.escape(code) + r'\b', name, text)
            
            text = re.sub(r'\s+', ' ', text).strip()
            response_data['risposta'] = text
        
        if 'dati' in response_data and response_data['dati']:
            dati = response_data['dati']
            
            if 'risultati' in dati:
                for risultato in dati['risultati']:
                    if 'prodotto' in risultato:
                        code = str(risultato['prodotto'])
                        name = self.code_to_name_product(code)
                        if name:
                            risultato['prodotto_nome'] = name
                            risultato['prodotto_codice'] = code
                            risultato['prodotto'] = name
                    
                    if 'cliente' in risultato and risultato['cliente'] != 'tutti':
                        code = str(risultato['cliente'])
                        name = self.code_to_name_client(code)
                        if name:
                            risultato['cliente_nome'] = name
                            risultato['cliente_codice'] = code
                            risultato['cliente'] = name
        
        return response_data
    
    def get_all_products(self) -> List[Dict[str, str]]:
        """Lista tutti i prodotti"""
        return [
            {'nome': name, 'codice': code}
            for code, name in self.product_code_to_name.items()
        ]
    
    def get_all_clients(self) -> List[Dict[str, str]]:
        """Lista tutti i clienti"""
        return [
            {'nome': name, 'codice': code}
            for code, name in self.client_code_to_name.items()
        ]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    mapper = NameCodeMapper()
    
    print("\n=== TEST: Nome parziale prodotto ===")
    text = "Quanto ho venduto di Osvego 3500 a gennaio 2024?"
    translated, meta = mapper.translate_names_to_codes(text)
    print(f"Originale: {text}")
    print(f"Tradotto: {translated}")
    print(f"Prodotti: {meta['prodotti_tradotti']}")
    print(f"Clienti: {meta['clienti_tradotti']}")
