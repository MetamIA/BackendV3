"""
Flask App - Entity Extraction + Translation
GPT estrae entità → NameCodeMapper traduce SOLO quelle
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import json

load_dotenv()

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from config.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, OPENAI_API_KEY
from modules.conversation_manager import get_conversation_manager
from modules.llm_orchestrator_enhanced import LLMOrchestratorEnhanced
from modules.prediction_manager_enhanced import PredictionManagerEnhanced
from modules.name_code_mapper import NameCodeMapper
from modules.trends_analyzer import TrendsAnalyzer
from modules.rag_system import RAGSystem
from modules.statistics_agent import StatisticsAgent
from openai import OpenAI

app = Flask(__name__)
CORS(app)

print("\n" + "="*60)
print("🚀 SISTEMA COMPLETO - CONVERSATIONAL STATISTICS")
print("="*60)

if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY non trovata!")
    sys.exit(1)

try:
    name_mapper = NameCodeMapper()
    print("✅ NameCodeMapper inizializzato")
    
    conversation_manager = get_conversation_manager(max_history=10)
    print("✅ Conversation Manager inizializzato")
    
    prediction_manager = PredictionManagerEnhanced()
    print("✅ Prediction Manager Enhanced inizializzato")
    
    # Statistics Agent (nuovo!)
    try:
        statistics_agent = StatisticsAgent()
        print("✅ Statistics Agent inizializzato (LLM conversazionale)")
    except Exception as e:
        print(f"⚠️  Statistics Agent non disponibile: {e}")
        statistics_agent = None
    
    # Trends Analyzer (con CSV per descrizioni prodotti)
    try:
        trends_analyzer = TrendsAnalyzer(df_predictions=prediction_manager.df_predictions)
        print("✅ Trends Analyzer inizializzato")
    except Exception as e:
        print(f"⚠️  Trends Analyzer non disponibile: {e}")
        trends_analyzer = None
    
    # RAG System
    try:
        rag_system = RAGSystem(openai_api_key=OPENAI_API_KEY)
        if rag_system.available:
            print("✅ RAG System inizializzato")
        else:
            print("⚠️  RAG System non disponibile (dipendenze mancanti)")
            rag_system = None
    except Exception as e:
        print(f"⚠️  RAG System errore: {e}")
        rag_system = None
    
    orchestrator = LLMOrchestratorEnhanced()
    print("✅ LLM Orchestrator Enhanced inizializzato")
    
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI Client inizializzato")
    
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("="*60)
print(f"✅ Sistema pronto su http://{FLASK_HOST}:{FLASK_PORT}")
print("   💡 GPT estrae entità, NameCodeMapper traduce")
print("="*60 + "\n")


@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint principale con entity extraction GPT"""
    
    try:
        data = request.json
        user_message_original = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not user_message_original:
            return jsonify({'tipo': 'error', 'error': 'Messaggio vuoto'}), 400
        
        print(f"\n{'='*60}")
        print(f"📩 Messaggio: {user_message_original}")
        
        # Storico
        conversation_history = conversation_manager.get_conversation_history(
            session_id, format_for_openai=True
        )
        
        # STEP 1: GPT analizza ed estrae entità
        plan = orchestrator.analyze_and_plan(user_message_original, conversation_history)
        
        # Chat generica
        if plan['tipo_richiesta'] == 'chat':
            risposta = plan.get('risposta_diretta', 'Ciao! Come posso aiutarti?')
            response_data = {
                'tipo': 'chat',
                'risposta': risposta,
                'session_id': session_id
            }
        
        # Prediction o Multi-module
        elif plan['tipo_richiesta'] in ['prediction', 'multi_module']:
            
            results = {}
            
            # TRENDS MODULE
            if plan['moduli'].get('trends', {}).get('attivo'):
                trends_module = plan['moduli']['trends']
                trends_input = trends_module.get('input', {})
                
                print(f"\n📈 Esecuzione: Trends Analysis")
                
                # Traduci prodotti
                if plan.get('entita', {}).get('prodotti'):
                    prodotti_codici = []
                    for prod_name in plan['entita']['prodotti']:
                        code_exact = None
                        for full_name_lower in name_mapper.product_names_lower:
                            if prod_name.lower() in full_name_lower:
                                full_name = name_mapper.product_names[name_mapper.product_names_lower.index(full_name_lower)]
                                code_exact = name_mapper.product_name_to_code.get(full_name_lower)
                                print(f"   ✓ Prodotto: '{prod_name}' → '{full_name}' (codice {code_exact})")
                                break
                        
                        if code_exact:
                            prodotti_codici.append(code_exact)
                    
                    if prodotti_codici:
                        trends_input['prodotti'] = prodotti_codici
                
                # Esegui trends
                if trends_analyzer:
                    prodotti = trends_input.get('prodotti', [])
                    periodo = trends_input.get('periodo', {'mese': 1, 'anno': 2024})
                    
                    trends_results = trends_analyzer.analyze_products_trends(prodotti, periodo)
                    
                    results['trends'] = {
                        'operazione': 'trends_analysis',
                        'risultati': [{
                            'formatted_trends': trends_analyzer.format_trends_for_llm(trends_results)
                        }],
                        'num_risultati': 1
                    }
                else:
                    results['trends'] = {
                        'operazione': 'trends_analysis',
                        'risultati': [{'messaggio': 'Trends Analyzer non disponibile'}],
                        'num_risultati': 1
                    }
            
            # PREDICTION MODULE
            if plan['moduli'].get('prediction', {}).get('attivo'):
                pred_module = plan['moduli']['prediction']
                operazione = pred_module.get('operazione', 'predizione_singola')
                input_data = pred_module.get('input', {})
                
                # STEP 2: Traduci SOLO le entità estratte da GPT
                if plan.get('entita'):
                    entita = plan['entita']
                    
                    # Traduci prodotti estratti
                    if entita.get('prodotti') and entita['prodotti'] != ['*']:
                        prodotti_codici = []
                        for prod_name in entita['prodotti']:
                            # Prima prova match ESATTO case-insensitive
                            code_exact = None
                            for full_name_lower in name_mapper.product_names_lower:
                                if prod_name.lower() in full_name_lower:
                                    # Trovato match parziale esatto
                                    full_name = name_mapper.product_names[name_mapper.product_names_lower.index(full_name_lower)]
                                    code_exact = name_mapper.product_name_to_code.get(full_name_lower)
                                    print(f"   ✓ Prodotto match esatto: '{prod_name}' → '{full_name}' (codice {code_exact})")
                                    break
                            
                            if code_exact:
                                prodotti_codici.append(code_exact)
                            else:
                                # Fallback: fuzzy match
                                matches = name_mapper.fuzzy_match_product(prod_name, threshold=0.6)
                                if matches:
                                    best_match = matches[0]
                                    codice = best_match[1]
                                    prodotti_codici.append(codice)
                                    print(f"   ✓ Prodotto fuzzy: '{prod_name}' → '{best_match[0]}' (codice {codice}, score {best_match[2]:.2f})")
                        
                        if prodotti_codici:
                            input_data['prodotti'] = prodotti_codici
                    
                    # Traduci clienti estratti
                    if entita.get('clienti') and entita['clienti']:
                        clienti_codici = []
                        for client_name in entita['clienti']:
                            # Cerca match per nome estratto
                            matches = name_mapper.fuzzy_match_client(client_name, threshold=0.6)
                            if matches:
                                best_match = matches[0]
                                codice = best_match[1]
                                clienti_codici.append(codice)
                                print(f"   ✓ Cliente: '{client_name}' → codice {codice}")
                        
                        if clienti_codici:
                            input_data['clienti'] = clienti_codici
                    else:
                        # Clienti non specificati → aggregazione
                        print(f"   → Clienti non specificati → aggregazione su tutti")
                
                print(f"\n📊 Esecuzione: {operazione}")
                print(f"   Input finale: {json.dumps(input_data, indent=6)}")
                
                # ESEGUI OPERAZIONE
                if operazione == 'trends_analysis':
                    # Analisi trends con Google Trends
                    if trends_analyzer:
                        prodotti = input_data.get('prodotti', [])
                        periodo = input_data.get('periodo', {'mese': 1, 'anno': 2024})
                        
                        trends_results = trends_analyzer.analyze_products_trends(prodotti, periodo)
                        
                        pred_results = {
                            'operazione': 'trends_analysis',
                            'risultati': [{
                                'prodotto': p,
                                'trends_data': trends_results.get(p, {})
                            } for p in prodotti],
                            'num_risultati': len(prodotti),
                            'formatted_trends': trends_analyzer.format_trends_for_llm(trends_results)
                        }
                    else:
                        pred_results = {
                            'operazione': 'trends_analysis',
                            'risultati': [{
                                'messaggio': 'Trends Analyzer non disponibile. Installa: pip install pytrends'
                            }],
                            'num_risultati': 1
                        }
                
                elif operazione == 'rag_query':
                    # Query su documenti aziendali
                    if rag_system and rag_system.available:
                        question = input_data.get('question', user_message_original)
                        
                        rag_results = rag_system.query_documents(question)
                        
                        pred_results = {
                            'operazione': 'rag_query',
                            'risultati': [{
                                'context': rag_results.get('context', ''),
                                'sources': rag_results.get('sources', []),
                                'n_chunks': rag_results.get('n_chunks', 0)
                            }],
                            'num_risultati': 1
                        }
                    else:
                        pred_results = {
                            'operazione': 'rag_query',
                            'risultati': [{
                                'messaggio': 'RAG System non disponibile. Installa: pip install docling chromadb sentence-transformers'
                            }],
                            'num_risultati': 1
                        }
                
                else:
                    # Operazioni statistiche o predizioni
                    
                    # Determina se è statistica (passato) o predizione (futuro)
                    is_statistics = operazione in ['somma', 'media', 'mediana', 'aggregazione_clienti']
                    
                    if is_statistics and statistics_agent:
                        # USA STATISTICS AGENT (LLM conversazionale)
                        # Passa la query ORIGINALE direttamente all'agent
                        # L'agent farà domande se necessario
                        
                        print(f"\n💬 Statistics Agent conversazionale")
                        print(f"   → Query originale passata all'agent")
                        
                        # Chiamata agent con query originale
                        agent_result = statistics_agent.query_statistics(
                            user_message_original,
                            conversation_history
                        )
                        
                        # Controlla se l'agent sta chiedendo chiarimenti
                        if agent_result.get('needs_clarification'):
                            # L'agent ha fatto una domanda!
                            response_data = {
                                'tipo': 'clarification',
                                'risposta': agent_result['risposta'],
                                'question': agent_result.get('question'),
                                'suggestions': agent_result.get('suggestions', []),
                                'session_id': session_id
                            }
                            
                            # Salva in memoria
                            conversation_manager.add_message(session_id, 'user', user_message_original)
                            conversation_manager.add_message(session_id, 'assistant', agent_result['risposta'])
                            
                            print(f"✅ Agent ha chiesto chiarimenti")
                            return jsonify(response_data)
                        
                        # Altrimenti ha una risposta finale
                        pred_results = {
                            'operazione': operazione,
                            'risultati': [{
                                'risposta_agent': agent_result['risposta'],
                                'tool_calls': agent_result['tool_calls'],
                                'iterations': agent_result.get('iterations', 0)
                            }],
                            'num_risultati': 1,
                            'agent_mode': True
                        }
                    
                    else:
                        # USA PREDICTION MANAGER (pandas + ML)
                        print(f"\n📊 Predizioni ML: {operazione}")
                        pred_results = prediction_manager.execute_operation(
                            operazione, input_data
                        )
                
                print(f"✅ Risultati: {pred_results.get('num_risultati', 0)} risultati")
                
                if pred_results.get('risultati'):
                    print(f"   Primo risultato: {json.dumps(pred_results['risultati'][0], indent=6)}")
                
                results['prediction'] = pred_results
            
            # COSTRUZIONE RISPOSTA DETERMINISTICA
            risposta_finale = build_multi_module_response(
                plan,
                results
            )
            
            response_data = {
                'tipo': 'prediction',
                'risposta': risposta_finale,
                'session_id': session_id,
                'operazione': 'multi_module' if len(results) > 1 else plan['moduli'].get('prediction', {}).get('operazione'),
                'dati': results
            }
        
        else:
            response_data = {
                'tipo': 'error',
                'error': 'Tipo richiesta non supportato'
            }
        
        # STEP 3: Traduzione codici → nomi nella risposta
        response_data = name_mapper.translate_response(response_data)
        
        # Salva memoria
        conversation_manager.add_message(session_id, 'user', user_message_original)
        conversation_manager.add_message(session_id, 'assistant', response_data['risposta'])
        
        print(f"\n✅ Risposta: {response_data['risposta'][:150]}...")
        print(f"{'='*60}\n")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({'tipo': 'error', 'error': str(e)}), 500


def build_multi_module_response(plan: dict, results: dict) -> str:
    """Costruisce risposta per richieste multi-modulo"""
    
    risposte_moduli = []
    
    # TRENDS
    if 'trends' in results and results['trends'].get('risultati'):
        trends_resp = build_trends_analysis_response(results['trends']['risultati'])
        risposte_moduli.append(trends_resp)
    
    # PREDICTION
    if 'prediction' in results and results['prediction'].get('risultati'):
        pred_data = results['prediction']
        
        # Controlla se è risposta da Statistics Agent
        if pred_data.get('agent_mode'):
            # Agent ha già generato risposta naturale
            risultati = pred_data['risultati']
            if risultati and 'risposta_agent' in risultati[0]:
                agent_resp = risultati[0]['risposta_agent']
                iterations = risultati[0].get('iterations', 0)
                
                resp = f"📊 Analisi Dati\n\n{agent_resp}"
                if iterations > 1:
                    resp += f"\n\n🔍 (Analisi completata in {iterations} iterazioni)"
                
                risposte_moduli.append(resp)
        else:
            # Risposta da Prediction Manager (template standard)
            operazione = pred_data.get('operazione', 'unknown')
            risultati = pred_data['risultati']
            
            if operazione == 'predizione_singola':
                pred_resp = build_predizione_singola_response(risultati)
            elif operazione == 'somma':
                pred_resp = build_somma_response(risultati)
            elif operazione == 'media':
                pred_resp = build_media_response(risultati)
            elif operazione == 'mediana':
                pred_resp = build_mediana_response(risultati)
            elif operazione == 'aggregazione_clienti':
                pred_resp = build_aggregazione_clienti_response(risultati)
            else:
                pred_resp = build_generic_response(risultati, operazione)
            
            risposte_moduli.append(pred_resp)
    
    # RAG
    if 'rag' in results and results['rag'].get('risultati'):
        rag_resp = build_rag_response(results['rag']['risultati'])
        risposte_moduli.append(rag_resp)
    
    # Concatena con separatore
    if len(risposte_moduli) > 1:
        return "\n\n" + "="*50 + "\n\n".join(risposte_moduli)
    elif len(risposte_moduli) == 1:
        return risposte_moduli[0]
    else:
        return "Non ho trovato dati per questa richiesta."


def build_deterministic_response(plan: dict, results: dict) -> str:
    """Costruisce risposta DETERMINISTICA con template Python"""
    
    if not results.get('prediction') or not results['prediction'].get('risultati'):
        return "Non ho trovato dati per questa richiesta. Verifica prodotto e periodo."
    
    pred_data = results['prediction']
    
    # Controlla se è risposta da Statistics Agent
    if pred_data.get('agent_mode'):
        risultati = pred_data['risultati']
        if risultati and 'risposta_agent' in risultati[0]:
            agent_resp = risultati[0]['risposta_agent']
            iterations = risultati[0].get('iterations', 0)
            
            resp = f"📊 Analisi Dati\n\n{agent_resp}"
            if iterations > 1:
                resp += f"\n\n🔍 (Analisi completata in {iterations} iterazioni)"
            
            return resp
    
    # Altrimenti usa template standard
    operazione = pred_data.get('operazione', 'unknown')
    risultati = pred_data['risultati']
    
    if not risultati:
        return "Non ho trovato dati per questa richiesta."
    
    # Template basato su operazione
    if operazione == 'predizione_singola':
        return build_predizione_singola_response(risultati)
    elif operazione == 'somma':
        return build_somma_response(risultati)
    elif operazione == 'media':
        return build_media_response(risultati)
    elif operazione == 'mediana':
        return build_mediana_response(risultati)
    elif operazione == 'aggregazione_clienti':
        return build_aggregazione_clienti_response(risultati)
    elif operazione == 'trends_analysis':
        return build_trends_analysis_response(risultati)
    elif operazione == 'rag_query':
        return build_rag_response(risultati)
    else:
        return build_generic_response(risultati, operazione)


def build_predizione_singola_response(risultati: list) -> str:
    """Template predizione singola"""
    res = risultati[0]
    
    prodotto = res.get('prodotto', 'Prodotto sconosciuto')
    cliente = res.get('cliente', 'Cliente sconosciuto')
    kg = res.get('kg_predetti', 0)
    tipo_dato = res.get('tipo_dato', 'unknown')
    confidenza = res.get('confidenza')
    
    risposta = f"📊 Predizione\n\n"
    risposta += f"Prodotto: {prodotto}\n"
    risposta += f"Cliente: {cliente}\n"
    risposta += f"Kg predetti: {kg:.2f} kg\n"
    
    if tipo_dato == 'storico':
        risposta += f"\nDato storico (fonte: CSV)"
    elif tipo_dato == 'predizione':
        risposta += f"\nPredizione ML"
        if confidenza is not None:
            risposta += f" (confidenza: {confidenza:.0%})"
    
    return risposta


def build_somma_response(risultati: list) -> str:
    """Template somma"""
    res = risultati[0]
    
    prodotto = res.get('prodotto', 'Prodotto sconosciuto')
    valore = res.get('valore', 0)
    num_periodi = res.get('num_periodi', 0)
    tipo_dato = res.get('tipo_dato', 'unknown')
    
    if res.get('clienti') == 'tutti':
        cliente_str = f"Tutti i clienti ({res.get('num_clienti', '?')} clienti)"
    else:
        cliente_str = f"Cliente: {res.get('cliente', 'N/A')}"
    
    risposta = f"📊 Somma Totale\n\n"
    risposta += f"Prodotto: {prodotto}\n"
    risposta += f"{cliente_str}\n"
    risposta += f"Totale: {valore:.2f} kg\n"
    
    if num_periodi > 0:
        risposta += f"Periodi analizzati: {num_periodi}\n"
    
    if tipo_dato == 'storico':
        risposta += f"\nDato storico (fonte: CSV)"
    elif tipo_dato == 'predizione':
        risposta += f"\nInclude predizioni ML"
    
    return risposta


def build_media_response(risultati: list) -> str:
    """Template media"""
    res = risultati[0]
    
    prodotto = res.get('prodotto', 'Prodotto sconosciuto')
    valore = res.get('valore', 0)
    num_periodi = res.get('num_periodi', 0)
    tipo_dato = res.get('tipo_dato', 'unknown')
    
    if res.get('clienti') == 'tutti':
        cliente_str = f"Tutti i clienti ({res.get('num_clienti', '?')} clienti)"
    else:
        cliente_str = f"Cliente: {res.get('cliente', 'N/A')}"
    
    risposta = f"📊 Media\n\n"
    risposta += f"Prodotto: {prodotto}\n"
    risposta += f"{cliente_str}\n"
    risposta += f"Media: {valore:.2f} kg/periodo\n"
    
    if num_periodi > 0:
        risposta += f"Periodi analizzati: {num_periodi}\n"
    
    if tipo_dato == 'storico':
        risposta += f"\nDato storico (fonte: CSV)"
    
    return risposta


def build_mediana_response(risultati: list) -> str:
    """Template mediana"""
    res = risultati[0]
    
    prodotto = res.get('prodotto', 'Prodotto sconosciuto')
    valore = res.get('valore', 0)
    num_periodi = res.get('num_periodi', 0)
    
    risposta = f"📊 Mediana\n\n"
    risposta += f"Prodotto: {prodotto}\n"
    risposta += f"Mediana: {valore:.2f} kg\n"
    
    if num_periodi > 0:
        risposta += f"Periodi analizzati: {num_periodi}\n"
    
    risposta += f"\nDato storico (fonte: CSV)"
    
    return risposta


def build_aggregazione_clienti_response(risultati: list) -> str:
    """Template aggregazione clienti"""
    res = risultati[0]
    
    prodotto = res.get('prodotto', 'Prodotto sconosciuto')
    valore = res.get('valore', 0)
    num_clienti = res.get('num_clienti', 0)
    tipo_dato = res.get('tipo_dato', 'unknown')
    periodo = res.get('periodo', {})
    
    risposta = f"📊 Vendite Totali\n\n"
    risposta += f"Prodotto: {prodotto}\n"
    
    if periodo:
        mese = periodo.get('mese')
        anno = periodo.get('anno')
        mesi = ['', 'Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno',
                'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre']
        if mese and anno:
            risposta += f"Periodo: {mesi[mese]} {anno}\n"
    
    risposta += f"Totale: {valore:.2f} kg\n"
    risposta += f"Clienti: {num_clienti}\n"
    
    if tipo_dato == 'storico':
        risposta += f"\nDato storico (fonte: CSV)"
    elif tipo_dato == 'predizione':
        risposta += f"\nPredizione ML"
    
    return risposta


def build_trends_analysis_response(risultati: list) -> str:
    """Template trends analysis con dati reali"""
    
    # Se ci sono dati formattati, usali
    if len(risultati) > 0 and risultati[0].get('formatted_trends'):
        return risultati[0]['formatted_trends']
    
    # Altrimenti costruisci risposta base
    res = risultati[0]
    
    if 'messaggio' in res:
        # Modulo non disponibile
        risposta = f"📈 Analisi Trends\n\n"
        risposta += f"{res['messaggio']}\n\n"
        risposta += f"ℹ️  Per abilitare questa funzionalità:\n"
        risposta += f"  pip install pytrends\n\n"
        risposta += f"Funzionalità attualmente disponibili:\n"
        risposta += f"  - Predizioni singole\n"
        risposta += f"  - Somme e medie\n"
        risposta += f"  - Aggregazioni clienti"
        return risposta
    
    prodotto = res.get('prodotto', 'Prodotto sconosciuto')
    trends_data = res.get('trends_data', {})
    
    risposta = f"📈 Analisi Trends\n\n"
    risposta += f"Prodotto: {prodotto}\n\n"
    
    if trends_data:
        descrizione = trends_data.get('descrizione', '')
        keywords = trends_data.get('keywords', [])
        
        risposta += f"Descrizione: {descrizione}\n"
        if keywords:
            risposta += f"Keywords analizzate: {', '.join(keywords)}\n\n"
        
        if trends_data.get('trends_data'):
            td = trends_data['trends_data']
            risposta += f"Periodo analisi: {td.get('timeframe', 'N/A')}\n\n"
            
            keywords_data = td.get('keywords_data', {})
            if keywords_data:
                risposta += "Risultati Google Trends:\n"
                for keyword, metrics in keywords_data.items():
                    trend_emoji = "📈" if metrics['trend'] == 'crescente' else "📉"
                    risposta += f"{trend_emoji} '{keyword}': interesse {metrics['media']:.0f}/100, "
                    risposta += f"{metrics['trend']} ({metrics['variazione']:+.1f}%)\n"
    else:
        risposta += "Nessun dato trends disponibile per questo prodotto."
    
    return risposta


def build_rag_response(risultati: list) -> str:
    """Template RAG query"""
    res = risultati[0]
    
    if 'messaggio' in res:
        # Sistema non disponibile
        risposta = f"📚 Query Documenti\n\n"
        risposta += f"{res['messaggio']}\n\n"
        risposta += f"ℹ️  Per abilitare questa funzionalità:\n"
        risposta += f"  pip install docling chromadb sentence-transformers"
        return risposta
    
    context = res.get('context', '')
    sources = res.get('sources', [])
    n_chunks = res.get('n_chunks', 0)
    
    risposta = f"📚 Informazioni dai Documenti\n\n"
    
    if context:
        risposta += f"Trovati {n_chunks} chunk rilevanti\n"
        risposta += f"Fonti: {', '.join(sources)}\n\n"
        risposta += "Contenuto:\n"
        risposta += context[:500]  # Prime 500 char
        if len(context) > 500:
            risposta += "\n\n[...testo troncato...]"
    else:
        risposta += "Nessun documento rilevante trovato per questa domanda.\n\n"
        risposta += "ℹ️  Assicurati di aver caricato documenti in data/documents/"
    
    return risposta


def build_generic_response(risultati: list, operazione: str) -> str:
    """Template generico"""
    res = risultati[0]
    
    risposta = f"📊 Operazione: {operazione}\n\n"
    
    if 'prodotto' in res:
        risposta += f"Prodotto: {res['prodotto']}\n"
    
    if 'cliente' in res:
        risposta += f"Cliente: {res['cliente']}\n"
    
    if 'valore' in res:
        risposta += f"Valore: {res['valore']:.2f} kg\n"
    elif 'kg_predetti' in res:
        risposta += f"Kg: {res['kg_predetti']:.2f} kg\n"
    
    return risposta


# Endpoint ausiliari
@app.route('/api/suggestions', methods=['POST'])
def suggestions():
    """Endpoint suggerimenti"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        tipo = data.get('type', 'product')
        
        if not query or len(query) < 2:
            return jsonify({'suggestions': []})
        
        if tipo == 'product':
            matches = name_mapper.fuzzy_match_product(query, threshold=0.4)
        else:
            matches = name_mapper.fuzzy_match_client(query, threshold=0.4)
        
        suggestions = [
            {'nome': name, 'codice': code, 'score': score}
            for name, code, score in matches
        ]
        
        return jsonify({'suggestions': suggestions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear_memory', methods=['POST'])
def clear_memory():
    """Cancella memoria"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        conversation_manager.clear_session(session_id)
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'version': 'entity-extraction',
        'features': [
            'gpt_entity_extraction',
            'deterministic_templates',
            'only_real_data'
        ]
    })


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
