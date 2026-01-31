"""
Test Suite per Statistics Agent con CSV-RAG
- Query semplici e complesse
- Verifica CONSISTENZA matematica (stessa domanda → stessi numeri)
- Conversazioni multi-turno
"""

import sys
import os
from typing import Dict, List, Any
import time
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from modules.statistics_agent import StatisticsAgent


class StatisticsAgentTester:
    """Test runner per Statistics Agent con verifica consistenza"""
    
    def __init__(self):
        """Inizializza tester"""
        print("\n" + "="*70)
        print("🧪 STATISTICS AGENT TEST SUITE - CSV-RAG")
        print("="*70)
        
        # Inizializza agent
        print("\n🔄 Inizializzazione Statistics Agent...")
        self.agent = StatisticsAgent(use_rag=True)
        
        if not self.agent.use_rag:
            print("⚠️  RAG non disponibile - alcuni test potrebbero fallire")
        
        self.results = []
        self.conversation_history = []
    
    def run_test(
        self,
        test_name: str,
        query: str,
        expected_keywords: List[str] = None,
        should_ask_clarification: bool = False,
        expected_numbers: List[float] = None
    ) -> Dict:
        """
        Esegue un singolo test
        
        Args:
            test_name: Nome del test
            query: Query da testare
            expected_keywords: Keywords attese nella risposta
            should_ask_clarification: Se True, ci aspettiamo domanda di chiarimento
            expected_numbers: Numeri attesi nella risposta (per verifica consistenza)
        """
        print("\n" + "-"*70)
        print(f"📝 TEST: {test_name}")
        print(f"Query: \"{query}\"")
        print("-"*70)
        
        start_time = time.time()
        
        try:
            # Esegui query
            result = self.agent.query_statistics(
                user_query=query,
                conversation_history=self.conversation_history.copy()
            )
            
            elapsed = time.time() - start_time
            
            risposta = result.get('risposta', '')
            needs_clarification = result.get('needs_clarification', False)
            iterations = result.get('iterations', 0)
            tool_calls = result.get('tool_calls', [])
            
            print(f"\n⏱️  Tempo: {elapsed:.2f}s | Iterazioni: {iterations}")
            print(f"💬 Risposta:\n{risposta}\n")
            
            # Verifica risultati
            passed = True
            errors = []
            
            # Check 1: Clarification
            if should_ask_clarification and not needs_clarification:
                passed = False
                errors.append("❌ Atteso chiarimento ma non ricevuto")
            elif not should_ask_clarification and needs_clarification:
                passed = False
                errors.append("❌ Chiarimento non atteso ma ricevuto")
            
            # Check 2: Keywords
            if expected_keywords and not needs_clarification:
                for keyword in expected_keywords:
                    if keyword.lower() not in risposta.lower():
                        passed = False
                        errors.append(f"❌ Keyword mancante: '{keyword}'")
            
            # Check 3: Estrai numeri dalla risposta
            extracted_numbers = self._extract_numbers(risposta)
            
            # Check 4: Verifica numeri attesi (se forniti)
            if expected_numbers and extracted_numbers and not needs_clarification:
                if len(extracted_numbers) != len(expected_numbers):
                    passed = False
                    errors.append(f"❌ Numero di valori diverso: attesi {len(expected_numbers)}, trovati {len(extracted_numbers)}")
                else:
                    for i, (expected, actual) in enumerate(zip(expected_numbers, extracted_numbers)):
                        if abs(expected - actual) > 0.01:  # Tolleranza 0.01
                            passed = False
                            errors.append(f"❌ Valore {i+1}: atteso {expected}, trovato {actual}")
            
            # Salva risultato
            test_result = {
                'test_name': test_name,
                'query': query,
                'risposta': risposta,
                'passed': passed,
                'errors': errors,
                'elapsed': elapsed,
                'iterations': iterations,
                'needs_clarification': needs_clarification,
                'extracted_numbers': extracted_numbers,
                'tool_calls_count': len(tool_calls)
            }
            
            self.results.append(test_result)
            
            # Print risultato
            if passed:
                print("✅ TEST PASSED")
            else:
                print("❌ TEST FAILED")
                for error in errors:
                    print(f"   {error}")
            
            if extracted_numbers:
                print(f"📊 Numeri estratti: {extracted_numbers}")
            
            return test_result
            
        except Exception as e:
            print(f"❌ ERRORE: {str(e)}")
            import traceback
            traceback.print_exc()
            
            test_result = {
                'test_name': test_name,
                'query': query,
                'passed': False,
                'errors': [f"Exception: {str(e)}"],
                'elapsed': time.time() - start_time
            }
            self.results.append(test_result)
            return test_result
    
    def run_consistency_test(
        self,
        test_name: str,
        query: str,
        repetitions: int = 3
    ) -> Dict:
        """
        Test di CONSISTENZA: stessa query ripetuta deve dare stessi numeri
        
        Args:
            test_name: Nome del test
            query: Query da ripetere
            repetitions: Numero di ripetizioni (default 3)
        """
        print("\n" + "="*70)
        print(f"🔄 TEST CONSISTENZA: {test_name}")
        print(f"Query: \"{query}\"")
        print(f"Ripetizioni: {repetitions}")
        print("="*70)
        
        all_numbers = []
        all_responses = []
        
        for i in range(repetitions):
            print(f"\n--- Tentativo {i+1}/{repetitions} ---")
            
            result = self.agent.query_statistics(
                user_query=query,
                conversation_history=[]  # Fresh conversation ogni volta
            )
            
            risposta = result.get('risposta', '')
            numbers = self._extract_numbers(risposta)
            
            all_numbers.append(numbers)
            all_responses.append(risposta)
            
            print(f"Numeri estratti: {numbers}")
            time.sleep(0.5)  # Pausa breve tra richieste
        
        # Verifica consistenza
        print("\n" + "-"*70)
        print("📊 VERIFICA CONSISTENZA")
        print("-"*70)
        
        passed = True
        errors = []
        
        if not all_numbers[0]:
            passed = False
            errors.append("❌ Nessun numero trovato nelle risposte")
        else:
            # Confronta tutti i set di numeri con il primo
            reference = all_numbers[0]
            
            for i, numbers in enumerate(all_numbers[1:], start=2):
                if len(numbers) != len(reference):
                    passed = False
                    errors.append(f"❌ Tentativo {i}: numero di valori diverso ({len(numbers)} vs {len(reference)})")
                else:
                    for j, (ref_num, test_num) in enumerate(zip(reference, numbers)):
                        if abs(ref_num - test_num) > 0.01:
                            passed = False
                            errors.append(f"❌ Tentativo {i}, valore {j+1}: {test_num} ≠ {ref_num}")
        
        # Risultato
        consistency_result = {
            'test_name': f"{test_name} (Consistency)",
            'query': query,
            'passed': passed,
            'errors': errors,
            'repetitions': repetitions,
            'all_numbers': all_numbers
        }
        
        self.results.append(consistency_result)
        
        if passed:
            print("✅ CONSISTENZA VERIFICATA - Tutti i numeri sono identici")
            print(f"   Valori: {all_numbers[0]}")
        else:
            print("❌ INCONSISTENZA RILEVATA")
            for error in errors:
                print(f"   {error}")
            print(f"\n   Tentativo 1: {all_numbers[0]}")
            for i, nums in enumerate(all_numbers[1:], start=2):
                print(f"   Tentativo {i}: {nums}")
        
        return consistency_result
    
    def run_conversation_test(
        self,
        test_name: str,
        messages: List[str],
        expected_flow: List[str] = None
    ) -> Dict:
        """
        Test conversazione multi-turno
        
        Args:
            test_name: Nome del test
            messages: Lista di messaggi da inviare in sequenza
            expected_flow: Tipo atteso per ogni risposta ('clarification', 'answer', 'error')
        """
        print("\n" + "="*70)
        print(f"💬 TEST CONVERSAZIONE: {test_name}")
        print("="*70)
        
        conversation = []
        passed = True
        errors = []
        
        for i, message in enumerate(messages):
            print(f"\n--- Turno {i+1}/{len(messages)} ---")
            print(f"User: {message}")
            
            result = self.agent.query_statistics(
                user_query=message,
                conversation_history=conversation.copy()
            )
            
            risposta = result.get('risposta', '')
            needs_clarification = result.get('needs_clarification', False)
            
            print(f"Agent: {risposta[:200]}...")
            
            # Aggiungi alla conversazione
            conversation.append({"role": "user", "content": message})
            conversation.append({"role": "assistant", "content": risposta})
            
            # Verifica flow atteso
            if expected_flow and i < len(expected_flow):
                expected = expected_flow[i]
                
                if expected == 'clarification' and not needs_clarification:
                    passed = False
                    errors.append(f"❌ Turno {i+1}: atteso chiarimento")
                elif expected == 'answer' and needs_clarification:
                    passed = False
                    errors.append(f"❌ Turno {i+1}: atteso risposta finale")
        
        conversation_result = {
            'test_name': f"{test_name} (Conversation)",
            'passed': passed,
            'errors': errors,
            'turns': len(messages),
            'conversation': conversation
        }
        
        self.results.append(conversation_result)
        
        if passed:
            print("\n✅ CONVERSAZIONE COMPLETATA CORRETTAMENTE")
        else:
            print("\n❌ CONVERSAZIONE FALLITA")
            for error in errors:
                print(f"   {error}")
        
        return conversation_result
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Estrae tutti i numeri da un testo (con virgole e punti)"""
        import re
        
        # Pattern per numeri (supporta: 1234, 1,234, 1234.56, 1.234,56)
        patterns = [
            r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\b',  # Con separatori migliaia
            r'\b\d+\.?\d*\b'  # Numeri semplici
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Normalizza (rimuovi virgole, converti)
                    normalized = match.replace(',', '')
                    numbers.append(float(normalized))
                except:
                    pass
        
        return numbers
    
    def generate_report(self) -> str:
        """Genera report finale"""
        print("\n" + "="*70)
        print("📊 REPORT FINALE")
        print("="*70)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get('passed', False))
        failed = total - passed
        
        print(f"\nTotale test: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success rate: {(passed/total*100):.1f}%")
        
        if failed > 0:
            print("\n❌ Test falliti:")
            for result in self.results:
                if not result.get('passed', False):
                    print(f"   - {result['test_name']}")
                    for error in result.get('errors', []):
                        print(f"     {error}")
        
        # Salva report JSON
        report_path = f"tests/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('tests', exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total': total,
                    'passed': passed,
                    'failed': failed,
                    'success_rate': passed/total*100
                },
                'results': self.results
            }, f, indent=2)
        
        print(f"\n📄 Report salvato: {report_path}")
        
        return report_path


def run_all_tests():
    """Esegue tutti i test"""
    
    tester = StatisticsAgentTester()
    
    print("\n" + "="*70)
    print("🚀 INIZIO TEST SUITE")
    print("="*70)
    
    # ========================================
    # SEZIONE 1: Test Query Semplici
    # ========================================
    print("\n" + "="*70)
    print("📋 SEZIONE 1: QUERY SEMPLICI")
    print("="*70)
    
    # Test 1: Query ambigua (dovrebbe chiedere chiarimenti)
    tester.run_test(
        test_name="Query ambigua - Osvego generico",
        query="vendite osvego",
        should_ask_clarification=True
    )
    
    # Test 2: Query specifica
    tester.run_test(
        test_name="Query specifica - Osvego 3500 nel 2024",
        query="totale kg venduti di Osvego 3500 nel 2024",
        expected_keywords=["Osvego", "2024", "kg"]
    )
    
    # Test 3: Query con periodo specifico
    tester.run_test(
        test_name="Query periodo - gennaio 2024",
        query="vendite Osvego 3500 a gennaio 2024",
        expected_keywords=["gennaio", "2024", "Osvego"]
    )
    
    # ========================================
    # SEZIONE 2: Test Consistenza (CRITICO!)
    # ========================================
    print("\n" + "="*70)
    print("🔄 SEZIONE 2: TEST CONSISTENZA MATEMATICA")
    print("="*70)
    
    # Test 4: Consistenza query totale
    tester.run_consistency_test(
        test_name="Consistenza totale vendite Osvego 3500 nel 2024",
        query="quanto ho venduto in totale di Osvego 3500 nel 2024?",
        repetitions=3
    )
    
    # Test 5: Consistenza media
    tester.run_consistency_test(
        test_name="Consistenza media vendite",
        query="qual è la media kg venduti di Osvego 3500 nel 2024?",
        repetitions=3
    )
    
    # ========================================
    # SEZIONE 3: Query Complesse (CSV-RAG)
    # ========================================
    print("\n" + "="*70)
    print("🎯 SEZIONE 3: QUERY COMPLESSE (RAG)")
    print("="*70)
    
    if tester.agent.use_rag:
        # Test 6: Query con filtro ricavo
        tester.run_test(
            test_name="Query complessa - Ricavi alti",
            query="quanti prodotti hanno ricavo netto maggiore di 100 euro nel 2024?",
            expected_keywords=["prodott", "ricavo"]
        )
        
        # Test 7: Query cluster
        tester.run_test(
            test_name="Query cluster - Clienti rank 3",
            query="vendite per clienti nel cluster rank 3",
            expected_keywords=["client", "cluster"]
        )
        
        # Test 8: Query sconti
        tester.run_test(
            test_name="Query sconti - Sconto merce alto",
            query="prodotti con sconto merce maggiore di 10 euro",
            expected_keywords=["sconto", "merce"]
        )
    else:
        print("⚠️  RAG non disponibile - test query complesse saltati")
    
    # ========================================
    # SEZIONE 4: Conversazioni Multi-Turno
    # ========================================
    print("\n" + "="*70)
    print("💬 SEZIONE 4: CONVERSAZIONI MULTI-TURNO")
    print("="*70)
    
    # Test 9: Conversazione con chiarimento
    tester.run_conversation_test(
        test_name="Conversazione chiarimento prodotto",
        messages=[
            "vendite osvego 2024",  # Ambiguo
            "Osvego 3500"  # Chiarimento
        ],
        expected_flow=['clarification', 'answer']
    )
    
    # Test 10: Conversazione raffinamento
    tester.run_conversation_test(
        test_name="Conversazione raffinamento periodo",
        messages=[
            "vendite Osvego 3500",  # Manca periodo
            "nel 2024",  # Specifica periodo
            "solo gennaio"  # Raffina
        ],
        expected_flow=['clarification', 'answer', 'answer']
    )
    
    # ========================================
    # SEZIONE 5: Edge Cases
    # ========================================
    print("\n" + "="*70)
    print("⚠️  SEZIONE 5: EDGE CASES")
    print("="*70)
    
    # Test 11: Prodotto inesistente
    tester.run_test(
        test_name="Edge case - Prodotto inesistente",
        query="vendite SuperProdottoChiamato12345XYZ nel 2024",
        should_ask_clarification=True  # È OK che chieda se intendeva qualcos'altro
    )
    
    # Test 12: Periodo futuro
    tester.run_test(
        test_name="Edge case - Periodo futuro",
        query="vendite Osvego 3500 a dicembre 2030",
        should_ask_clarification=True  # È OK che chieda un altro periodo
    )
    
    # ========================================
    # REPORT FINALE
    # ========================================
    tester.generate_report()
    
    print("\n" + "="*70)
    print("✅ TEST SUITE COMPLETATA")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
