"""
Script di Test Completo per il Sistema Modulare
Testa Query Parser, Prediction Manager e l'integrazione completa
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.query_parser import QueryParser
from modules.prediction_manager import PredictionManager

def test_query_parser():
    """Test del Modulo 1: Query Parser"""
    print("\n" + "="*60)
    print("TEST MODULO 1: QUERY PARSER")
    print("="*60)
    
    parser = QueryParser()
    
    test_cases = [
        ("Ciao come stai?", "chat"),
        ("Previsioni per prodotto 40000 cliente 393 a gennaio 2024", "query"),
        ("Vendite di prodotto 40000 per cliente 393 a marzo 2024", "query"),
        ("Dammi le vendite di prodotto 40000 per clienti 393 e 439 a giugno 2024", "query"),
    ]
    
    for message, expected_type in test_cases:
        print(f"\n📝 Input: {message}")
        result = parser.parse_message(message)
        print(f"   Tipo atteso: {expected_type}")
        print(f"   Tipo ricevuto: {result.get('tipo')}")
        
        if result.get('tipo') == 'query':
            print(f"   Combinazioni: {result.get('num_combinations')}")
        
        status = "✅" if result.get('tipo') == expected_type else "❌"
        print(f"   {status}")


def test_prediction_manager():
    """Test del Modulo 2: Prediction Manager"""
    print("\n" + "="*60)
    print("TEST MODULO 2: PREDICTION MANAGER")
    print("="*60)
    
    manager = PredictionManager()
    
    # Test con query combinations
    test_combinations = [
        {
            'prodotto': '40000',
            'cliente': '393',
            'periodo': {'mese': 1, 'anno': 2024}
        }
    ]
    
    print("\n📊 Test con combinazione:")
    print(f"   Prodotto: 40000")
    print(f"   Cliente: 393")
    print(f"   Periodo: 1/2024")
    
    result = manager.process_query_combinations(test_combinations)
    
    print(f"\n✅ Predizioni generate: {result['num_predizioni']}")
    print(f"\n💬 Risposta:\n{result['risposta']}")


def test_integration():
    """Test di integrazione completa"""
    print("\n" + "="*60)
    print("TEST INTEGRAZIONE COMPLETA")
    print("="*60)
    
    parser = QueryParser()
    manager = PredictionManager()
    
    # Simula una query completa
    user_message = "Dammi le previsioni per prodotto 40000 cliente 393 a gennaio 2024"
    
    print(f"\n📝 Query utente: {user_message}")
    
    # Step 1: Parse
    parse_result = parser.parse_message(user_message)
    print(f"\n1️⃣ Parsing completato")
    print(f"   Tipo: {parse_result.get('tipo')}")
    print(f"   Combinazioni: {parse_result.get('num_combinations')}")
    
    # Step 2: Prediction
    if parse_result.get('tipo') == 'query':
        prediction_result = manager.process_query_combinations(
            parse_result['query_combinations']
        )
        
        print(f"\n2️⃣ Predizioni completate")
        print(f"   Numero predizioni: {prediction_result['num_predizioni']}")
        
        print(f"\n3️⃣ Risposta finale:\n")
        print(prediction_result['risposta'])


def main():
    """Main test runner"""
    print("\n" + "="*60)
    print("🧪 SUITE DI TEST COMPLETA")
    print("="*60)
    
    try:
        # Test singoli moduli
        test_query_parser()
        test_prediction_manager()
        
        # Test integrazione
        test_integration()
        
        print("\n" + "="*60)
        print("✅ TUTTI I TEST COMPLETATI")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Errore durante i test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
