"""
Modulo 4: RAG System
Sistema di Retrieval-Augmented Generation per query su documenti aziendali
"""

import os
import sys
import json
import hashlib
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Docling imports
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("⚠️  Docling non disponibile. Installa con: pip install docling")

# Vector DB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB non disponibile. Installa con: pip install chromadb")

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers non disponibile. Installa con: pip install sentence-transformers")

# OpenAI per descrizione immagini
from openai import OpenAI
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import OPENAI_API_KEY, OPENAI_MODEL_RESPONSE


class FileTracker:
    """Gestisce il tracking dei file processati usando hash MD5"""
    
    def __init__(self, tracking_file: str = "data/processed_files.json"):
        self.tracking_file = tracking_file
        self.processed_files = self._load_tracking()
    
    def _load_tracking(self) -> Dict[str, Dict]:
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_tracking(self):
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        with open(self.tracking_file, 'w') as f:
            json.dump(self.processed_files, f, indent=2)
    
    def get_file_hash(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_file_processed(self, file_path: str) -> bool:
        current_hash = self.get_file_hash(file_path)
        file_name = os.path.basename(file_path)
        
        if file_name in self.processed_files:
            return self.processed_files[file_name]["hash"] == current_hash
        return False
    
    def mark_as_processed(self, file_path: str, metadata: Dict = None):
        file_hash = self.get_file_hash(file_path)
        file_name = os.path.basename(file_path)
        
        self.processed_files[file_name] = {
            "hash": file_hash,
            "path": file_path,
            "processed_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self._save_tracking()
    
    def get_new_files(self, file_paths: List[str]) -> List[str]:
        return [f for f in file_paths if not self.is_file_processed(f)]


class ImageDescriber:
    """Gestisce la descrizione delle immagini usando GPT-4 Vision"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def describe_image(self, image_data: bytes, context: str = "") -> str:
        try:
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            prompt = f"""Descrivi questa immagine in modo dettagliato e tecnico.
            Contesto del documento: {context if context else 'Documento aziendale'}
            
            Includi:
            - Tipo di contenuto (grafico, diagramma, foto, schema, tabella)
            - Elementi principali visibili
            - Dati o informazioni rilevanti (se presenti)
            - Relazioni tra elementi
            
            Rispondi in italiano in modo conciso."""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"[Immagine - descrizione non disponibile: {str(e)}]"


class DocumentProcessor:
    """Processa documenti usando Docling"""
    
    def __init__(self, openai_api_key: str):
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling non disponibile. Installa con: pip install docling")
        
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        
        self.image_describer = ImageDescriber(openai_api_key)
    
    def process_document(self, file_path: str, describe_images: bool = True) -> Dict[str, Any]:
        """Processa un documento e restituisce testo + metadata"""
        try:
            result = self.converter.convert(file_path)
            text_content = result.document.export_to_markdown()
            
            images_data = []
            if describe_images and file_path.lower().endswith('.pdf'):
                images_data = self._extract_and_describe_images(file_path, result)
            
            return {
                'text': text_content,
                'images': images_data,
                'file_path': file_path,
                'metadata': {
                    'file_name': os.path.basename(file_path),
                    'file_size': os.path.getsize(file_path),
                    'num_images': len(images_data)
                }
            }
            
        except Exception as e:
            print(f"Errore processamento {file_path}: {e}")
            return None
    
    def _extract_and_describe_images(self, pdf_path: str, result) -> List[Dict]:
        images = []
        try:
            if hasattr(result.document, 'pictures') and result.document.pictures:
                for idx, picture in enumerate(result.document.pictures[:5]):  # Max 5 immagini
                    if hasattr(picture, 'image') and picture.image:
                        # Converti PIL Image in bytes
                        from io import BytesIO
                        img_bytes = BytesIO()
                        picture.image.pil_image.save(img_bytes, format='JPEG')
                        img_bytes = img_bytes.getvalue()
                        
                        description = self.image_describer.describe_image(
                            img_bytes,
                            context=os.path.basename(pdf_path)
                        )
                        
                        images.append({
                            'index': idx,
                            'description': description,
                            'caption': getattr(picture, 'caption', ''),
                            'page': getattr(picture, 'page', -1)
                        })
        except Exception as e:
            print(f"  ⚠️  Errore estrazione immagini: {e}")
        
        return images


class VectorDatabase:
    """Gestisce il database vettoriale con ChromaDB"""
    
    def __init__(
        self,
        persist_directory: str = "data/chromadb",
        collection_name: str = "documents"
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB non disponibile. Installa con: pip install chromadb")
        
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers non disponibile")
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Usa modello multilingua italiano
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    def add_document(self, doc_data: Dict[str, Any], file_id: str):
        """Aggiunge un documento al database vettoriale"""
        if not doc_data:
            return
        
        chunks = []
        metadatas = []
        ids = []
        
        # Chunking del testo
        text_chunks = self._chunk_text(doc_data['text'])
        
        for i, chunk in enumerate(text_chunks):
            chunks.append(chunk)
            metadatas.append({
                'file_id': file_id,
                'file_name': doc_data['metadata']['file_name'],
                'chunk_type': 'text',
                'chunk_index': i,
                'source': doc_data['file_path']
            })
            ids.append(f"{file_id}_text_{i}")
        
        # Aggiungi immagini
        for i, img in enumerate(doc_data['images']):
            img_text = f"[IMMAGINE - Pagina {img['page']}]\n"
            if img['caption']:
                img_text += f"Didascalia: {img['caption']}\n"
            img_text += f"Descrizione: {img['description']}"
            
            chunks.append(img_text)
            metadatas.append({
                'file_id': file_id,
                'file_name': doc_data['metadata']['file_name'],
                'chunk_type': 'image',
                'chunk_index': i,
                'page': img['page'],
                'source': doc_data['file_path']
            })
            ids.append(f"{file_id}_image_{i}")
        
        if chunks:
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Divide il testo in chunks con overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """Cerca nel database vettoriale"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results


class RAGSystem:
    """Sistema RAG completo per query su documenti"""
    
    def __init__(
        self,
        openai_api_key: str,
        documents_dir: str = "data/documents"
    ):
        self.documents_dir = documents_dir
        self.openai_api_key = openai_api_key
        
        # Verifica disponibilità dipendenze
        if not (DOCLING_AVAILABLE and CHROMADB_AVAILABLE and EMBEDDINGS_AVAILABLE):
            print("⚠️  RAG System: alcune dipendenze mancanti")
            self.available = False
            return
        
        try:
            self.file_tracker = FileTracker()
            self.doc_processor = DocumentProcessor(openai_api_key)
            self.vector_db = VectorDatabase()
            self.openai_client = OpenAI(api_key=openai_api_key)
            self.available = True
            
            print("✅ Modulo 4: RAG System inizializzato")
        except Exception as e:
            print(f"⚠️  Errore inizializzazione RAG: {e}")
            self.available = False
    
    def ingest_documents(self, force_reprocess: bool = False):
        """Processa e indicizza documenti"""
        if not self.available:
            return
        
        supported_extensions = ['.pdf', '.docx', '.md', '.txt']
        all_files = []
        
        for ext in supported_extensions:
            all_files.extend(Path(self.documents_dir).rglob(f"*{ext}"))
        
        all_files = [str(f) for f in all_files]
        
        if not all_files:
            print(f"📂 Nessun documento in {self.documents_dir}")
            return
        
        if not force_reprocess:
            files_to_process = self.file_tracker.get_new_files(all_files)
        else:
            files_to_process = all_files
        
        if not files_to_process:
            print("✓ Tutti i documenti sono già indicizzati")
            return
        
        print(f"\n📚 Processamento {len(files_to_process)} documenti...")
        
        for i, file_path in enumerate(files_to_process, 1):
            print(f"[{i}/{len(files_to_process)}] {os.path.basename(file_path)}")
            
            try:
                doc_data = self.doc_processor.process_document(
                    file_path,
                    describe_images=file_path.lower().endswith('.pdf')
                )
                
                if doc_data:
                    file_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
                    self.vector_db.add_document(doc_data, file_id)
                    self.file_tracker.mark_as_processed(file_path, doc_data['metadata'])
                    print(f"  ✓ Indicizzato")
                
            except Exception as e:
                print(f"  ✗ Errore: {e}")
    
    def query_documents(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """Cerca informazioni nei documenti"""
        if not self.available:
            return {
                'answer': '',
                'sources': [],
                'context': ''
            }
        
        try:
            search_results = self.vector_db.search(question, n_results)
            
            if not search_results['documents'][0]:
                return {
                    'answer': '',
                    'sources': [],
                    'context': ''
                }
            
            context_chunks = search_results['documents'][0]
            context_metadatas = search_results['metadatas'][0]
            
            context = "\n\n---\n\n".join([
                f"[{meta['file_name']}]\n{doc}"
                for doc, meta in zip(context_chunks, context_metadatas)
            ])
            
            sources = list(set([meta['file_name'] for meta in context_metadatas]))
            
            return {
                'answer': '',  # GPT genererà risposta nel Prediction Manager
                'sources': sources,
                'context': context,
                'n_chunks': len(context_chunks)
            }
            
        except Exception as e:
            print(f"Errore query RAG: {e}")
            return {
                'answer': '',
                'sources': [],
                'context': ''
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiche sistema RAG"""
        if not self.available:
            return {'available': False}
        
        try:
            return {
                'available': True,
                'total_chunks': self.vector_db.collection.count(),
                'processed_files': len(self.file_tracker.processed_files),
                'files_list': list(self.file_tracker.processed_files.keys())
            }
        except:
            return {'available': False}


if __name__ == "__main__":
    # Test standalone
    from dotenv import load_dotenv
    load_dotenv()
    
    rag = RAGSystem(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        documents_dir="data/documents"
    )
    
    if rag.available:
        rag.ingest_documents()
        result = rag.query_documents("Quali sono le policy aziendali?")
        print(f"\nRisultato: {result}")
