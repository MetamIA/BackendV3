"""
Script di test interattivo per il Query Parser
"""

import os
import json
from dotenv import load_dotenv
from query_parser import QueryParser

# Carica le variabili d'ambiente dal file .env
load_dotenv()

def test_examples():
    """Test con esempi predefiniti"""
    
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        print("⚠️  OPENAI_API_KEY non trovata! Crea un file .env con la tua chiave API.")
        return
    
    parser = QueryParser(API_KEY)
    
    # Esempi di test
    test_cases = [
        # Chat normale
        "Ciao! Come va?",
        "Che tempo fa oggi?",
        "Grazie mille per l'aiuto!",
        
        # Query semplici
        "Previsioni per pasta cliente Rossi a gennaio 2025",
        "Vendite prodotto X cliente ABC a dicembre 2024",
        
        # Query con prodotto cartesiano
        "Dammi le previsioni per latte e burro per i clienti Bianchi e Verdi a marzo 2025",
        "Analisi di pasta, riso e olio per Rossi, Verdi e Bianchi nel mese di giugno 2025",
        
        # Query con formati diversi
        "Quanto venderemo di pasta a Rossi il prossimo gennaio?",
        "Cliente Verdi prodotto latte marzo 2025",
    ]
    
    print("=" * 80)
    print("🧪 TEST DEL QUERY PARSER")
    print("=" * 80)
    print()
    
    for i, test_message in enumerate(test_cases, 1):
        print(f"📝 Test {i}/{len(test_cases)}")
        print(f"Input: {test_message}")
        print()
        
        result = parser.parse_message(test_message)
        
        # Stampa formattata del risultato
        print(f"Tipo: {result.get('tipo', 'N/A')}")
        
        if result.get('tipo') == 'chat':
            print(f"Risposta: {result.get('risposta_chat', 'N/A')}")
        
        elif result.get('tipo') == 'query':
            dati = result.get('dati', {})
            print(f"Prodotti: {dati.get('prodotti', [])}")
            print(f"Clienti: {dati.get('clienti', [])}")
            print(f"Periodo: {dati.get('periodo', {})}")
            print(f"Numero combinazioni: {result.get('num_combinations', 0)}")
            
            if result.get('query_combinations'):
                print("\n🔄 Combinazioni generate:")
                for j, combo in enumerate(result['query_combinations'], 1):
                    print(f"  {j}. {combo['prodotto']} × {combo['cliente']} × {combo['periodo']['mese']}/{combo['periodo']['anno']}")
        
        print()
        print("-" * 80)
        print()


def interactive_mode():
    """Modalità interattiva per testare il parser"""
    
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        print("⚠️  OPENAI_API_KEY non trovata! Crea un file .env con la tua chiave API.")
        return
    
    parser = QueryParser(API_KEY)
    
    print("=" * 80)
    print("💬 MODALITÀ INTERATTIVA")
    print("=" * 80)
    print("Scrivi un messaggio per testare il parser (o 'exit' per uscire)")
    print()
    
    while True:
        user_input = input("Tu: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'esci']:
            print("👋 Arrivederci!")
            break
        
        if not user_input:
            continue
        
        result = parser.parse_message(user_input)
        
        print()
        if result.get('tipo') == 'chat':
            print(f"🤖 Bot: {result.get('risposta_chat', 'N/A')}")
        
        elif result.get('tipo') == 'query':
            print(f"🔍 Query identificata!")
            dati = result.get('dati', {})
            print(f"   Prodotti: {', '.join(dati.get('prodotti', []))}")
            print(f"   Clienti: {', '.join(dati.get('clienti', []))}")
            periodo = dati.get('periodo', {})
            print(f"   Periodo: {periodo.get('mese')}/{periodo.get('anno')}")
            print(f"   Combinazioni: {result.get('num_combinations', 0)}")
        
        elif result.get('tipo') == 'error':
            print(f"❌ Errore: {result.get('errore', 'Errore sconosciuto')}")
        
        print()


def main():
    """Menu principale"""
    print()
    print("🎯 TEST QUERY PARSER - Demand Forecasting")
    print()
    print("Seleziona modalità:")
    print("1. Test con esempi predefiniti")
    print("2. Modalità interattiva")
    print("3. Esci")
    print()
    
    choice = input("Scelta (1-3): ").strip()
    
    if choice == '1':
        test_examples()
    elif choice == '2':
        interactive_mode()
    elif choice == '3':
        print("👋 Arrivederci!")
    else:
        print("❌ Scelta non valida")


if __name__ == "__main__":
    main()
