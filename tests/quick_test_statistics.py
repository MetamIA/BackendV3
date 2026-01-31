"""
Quick Test - Verifica rapida funzionamento Statistics Agent
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from modules.statistics_agent import StatisticsAgent


def quick_test():
    """Test rapido con poche query essenziali"""
    
    print("\n" + "="*60)
    print("⚡ QUICK TEST - Statistics Agent")
    print("="*60)
    
    # Init agent
    print("\n🔄 Inizializzazione...")
    agent = StatisticsAgent(use_rag=True)
    
    if not agent.use_rag:
        print("⚠️  RAG non disponibile!")
        return
    
    print("✅ Agent pronto\n")
    
    # Test queries
    queries = [
        "vendite Osvego 3500 nel 2024",
        "media kg venduti di Osvego 3500 nel 2024",
        "prodotti con ricavo netto maggiore di 100 euro"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print("="*60)
        
        result = agent.query_statistics(query)
        
        risposta = result.get('risposta', '')
        iterations = result.get('iterations', 0)
        
        print(f"\n💬 Risposta ({iterations} iterazioni):")
        print(risposta)
        
        # Verifica tool calls
        tool_calls = result.get('tool_calls', [])
        if tool_calls:
            print(f"\n🔧 Tool calls: {len(tool_calls)}")
            for tc in tool_calls:
                print(f"   - {tc['function']}")
    
    print("\n" + "="*60)
    print("✅ Quick test completato")
    print("="*60)


if __name__ == "__main__":
    quick_test()
