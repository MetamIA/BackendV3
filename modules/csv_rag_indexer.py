"""
CSV RAG Indexer
Indicizza il CSV predictions.csv in ChromaDB per query semantiche
"""

import pandas as pd
import json
from typing import List, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PREDICTIONS_CSV

# ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB non disponibile. Installa: pip install chromadb")

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers non disponibile")


class CSVRAGIndexer:
    """
    Indicizza CSV in ChromaDB per retrieval semantico
    """
    
    def __init__(
        self,
        csv_path: str = str(PREDICTIONS_CSV),
        db_path: str = "data/csv_chroma_db",
        collection_name: str = "predictions_csv"
    ):
        """Inizializza indexer"""
        
        if not (CHROMADB_AVAILABLE and EMBEDDINGS_AVAILABLE):
            print("❌ CSV RAG richiede chromadb e sentence-transformers")
            self.available = False
            return
        
        self.csv_path = csv_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.available = True
        
        # Embedding model
        print("🔄 Caricamento embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ CSV RAG Indexer inizializzato")
        print(f"   Collection: {collection_name}")
        print(f"   DB path: {db_path}")
    
    def index_csv(self, force_reindex: bool = False, sample_size: int = None):
        """
        Indicizza CSV in ChromaDB
        
        Args:
            force_reindex: Se True, cancella e reindicizza tutto
            sample_size: Se specificato, indicizza solo N righe (per test)
        """
        
        if not self.available:
            return
        
        # Check se già indicizzato
        current_count = self.collection.count()
        
        if current_count > 0 and not force_reindex:
            print(f"✓ CSV già indicizzato ({current_count} chunks)")
            return
        
        if force_reindex and current_count > 0:
            print(f"🔄 Cancello {current_count} chunks esistenti...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Carica CSV
        print(f"📂 Caricamento CSV: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        if sample_size:
            df = df.head(sample_size)
            print(f"   📊 Modalità test: {len(df)} righe")
        else:
            print(f"   📊 Righe totali: {len(df)}")
        
        # Converti ogni riga in testo "searchable"
        print("🔄 Conversione righe in testo...")
        texts = []
        metadatas = []
        ids = []
        
        for idx, row in df.iterrows():
            # Crea testo searchable con tutte le colonne rilevanti
            text_parts = []
            
            # Prodotto e cliente (nomi leggibili)
            text_parts.append(f"Prodotto: {row['Descrizione_Prodotto']}")
            text_parts.append(f"Cliente: {row['Descrizione_Cliente']}")
            
            # Periodo
            mesi = ['', 'gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 'giugno',
                    'luglio', 'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre']
            mese_nome = mesi[row['Periodo']] if row['Periodo'] <= 12 else str(row['Periodo'])
            text_parts.append(f"Periodo: {mese_nome} {row['Esercizio']}")
            
            # Kg venduti
            text_parts.append(f"Kg venduti: {row['Kg_Venduti_Predetti']:.2f} kg")
            
            # Ricavi e sconti
            if pd.notna(row['Ricavo_Netto']):
                text_parts.append(f"Ricavo netto: €{row['Ricavo_Netto']:.2f}")
            if pd.notna(row['Sconto_Merce']) and row['Sconto_Merce'] > 0:
                text_parts.append(f"Sconto merce: €{row['Sconto_Merce']:.2f}")
            
            # Cluster
            if pd.notna(row['Cluster_Prodotto_Label']):
                text_parts.append(f"Cluster prodotto: {row['Cluster_Prodotto_Label']}")
            if pd.notna(row['Cluster_Cliente_Label']):
                text_parts.append(f"Cluster cliente: {row['Cluster_Cliente_Label']}")
            
            text = ". ".join(text_parts)
            texts.append(text)
            
            # Metadata (tutti i campi per calcoli)
            metadata = {
                'esercizio': int(row['Esercizio']),
                'periodo': int(row['Periodo']),
                'prodotto_codice': str(row['Prodotto']),
                'prodotto_nome': str(row['Descrizione_Prodotto']),
                'cliente_codice': str(row['Cliente']),
                'cliente_nome': str(row['Descrizione_Cliente']),
                'kg_venduti': float(row['Kg_Venduti_Predetti']),
                'ricavo_netto': float(row['Ricavo_Netto']) if pd.notna(row['Ricavo_Netto']) else 0.0,
                'sconto_merce': float(row['Sconto_Merce']) if pd.notna(row['Sconto_Merce']) else 0.0,
                'cluster_prodotto': str(row['Cluster_Prodotto_Label']) if pd.notna(row['Cluster_Prodotto_Label']) else '',
                'cluster_cliente': str(row['Cluster_Cliente_Label']) if pd.notna(row['Cluster_Cliente_Label']) else ''
            }
            metadatas.append(metadata)
            
            ids.append(f"row_{idx}")
        
        # Genera embeddings
        print(f"🧠 Generazione embeddings per {len(texts)} righe...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Inserisci in ChromaDB (a batch)
        print("💾 Inserimento in ChromaDB...")
        batch_size = 1000
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            
            self.collection.add(
                embeddings=embeddings[i:end_idx].tolist(),
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            
            print(f"   ✓ {end_idx}/{len(texts)} righe indicizzate")
        
        print(f"✅ CSV indicizzato: {len(texts)} righe")
    
    def search(
        self,
        query: str,
        n_results: int = 50,
        filters: Dict = None
    ) -> Dict[str, Any]:
        """
        Cerca nel CSV con query semantica
        
        Args:
            query: Query testuale naturale
            n_results: Numero risultati da ritornare
            filters: Filtri metadata (es. {"esercizio": 2024})
        
        Returns:
            Dict con risultati
        """
        
        if not self.available:
            return {'documents': [], 'metadatas': [], 'distances': []}
        
        # Genera embedding query
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=filters if filters else None
        )
        
        return results
    
    def get_stats(self) -> Dict:
        """Statistiche indexer"""
        if not self.available:
            return {'available': False}
        
        return {
            'available': True,
            'total_chunks': self.collection.count(),
            'collection_name': self.collection_name
        }


if __name__ == "__main__":
    # Test indexing
    from dotenv import load_dotenv
    load_dotenv()
    
    indexer = CSVRAGIndexer()
    
    if indexer.available:
        # Indicizza (test con prime 1000 righe)
        indexer.index_csv(force_reindex=True, sample_size=1000)
        
        # Test search
        print("\n=== TEST SEARCH ===")
        results = indexer.search("vendite osvego gennaio 2024", n_results=5)
        
        print(f"\nTrovati {len(results['documents'][0])} risultati:")
        for doc, meta in zip(results['documents'][0][:3], results['metadatas'][0][:3]):
            print(f"\n{doc}")
            print(f"   Metadata: {meta}")
