"""
Conversation Manager - Gestisce memoria e storico conversazioni
"""

import json
from typing import List, Dict, Optional
from datetime import datetime
import hashlib


class ConversationManager:
    """
    Gestisce lo storico delle conversazioni per mantenere memoria
    """
    
    def __init__(self, max_history: int = 10):
        """
        Inizializza il manager
        
        Args:
            max_history: Numero massimo di messaggi da tenere in memoria
        """
        self.max_history = max_history
        # Dict: session_id -> list of messages
        self.conversations = {}
    
    def get_or_create_session(self, session_id: str) -> List[Dict]:
        """
        Ottiene o crea una nuova sessione di conversazione
        
        Args:
            session_id: ID univoco della sessione
            
        Returns:
            Lista di messaggi della conversazione
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        return self.conversations[session_id]
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        Aggiunge un messaggio alla conversazione
        
        Args:
            session_id: ID sessione
            role: 'user' o 'assistant'
            content: Contenuto del messaggio
            metadata: Metadati opzionali (moduli attivati, etc.)
        """
        conversation = self.get_or_create_session(session_id)
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            message['metadata'] = metadata
        
        conversation.append(message)
        
        # Mantieni solo ultimi N messaggi
        if len(conversation) > self.max_history:
            # Mantieni sempre il primo messaggio (system prompt) se presente
            if conversation[0].get('role') == 'system':
                self.conversations[session_id] = [conversation[0]] + conversation[-(self.max_history-1):]
            else:
                self.conversations[session_id] = conversation[-self.max_history:]
    
    def get_conversation_history(
        self, 
        session_id: str,
        format_for_openai: bool = True
    ) -> List[Dict]:
        """
        Ottiene lo storico della conversazione
        
        Args:
            session_id: ID sessione
            format_for_openai: Se True, formatta per API OpenAI
            
        Returns:
            Lista di messaggi
        """
        conversation = self.get_or_create_session(session_id)
        
        if format_for_openai:
            # Formato per OpenAI: solo role e content
            return [
                {'role': msg['role'], 'content': msg['content']}
                for msg in conversation
            ]
        
        return conversation
    
    def get_context_summary(self, session_id: str) -> str:
        """
        Genera un riassunto del contesto della conversazione
        
        Args:
            session_id: ID sessione
            
        Returns:
            Stringa con contesto recente
        """
        conversation = self.get_or_create_session(session_id)
        
        if not conversation:
            return ""
        
        # Ultimi 3 scambi
        recent = conversation[-6:] if len(conversation) >= 6 else conversation
        
        context_parts = []
        for msg in recent:
            role_label = "Tu" if msg['role'] == 'user' else "Assistente"
            context_parts.append(f"{role_label}: {msg['content'][:100]}...")
        
        return "\n".join(context_parts)
    
    def clear_session(self, session_id: str):
        """Cancella una sessione"""
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    def get_active_sessions(self) -> List[str]:
        """Ottiene lista delle sessioni attive"""
        return list(self.conversations.keys())
    
    def generate_session_id(self, user_identifier: str = None) -> str:
        """
        Genera un ID sessione univoco
        
        Args:
            user_identifier: Identificativo opzionale dell'utente
            
        Returns:
            Session ID univoco
        """
        timestamp = datetime.now().isoformat()
        data = f"{user_identifier or 'anonymous'}_{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:16]


# Singleton globale per l'app
_conversation_manager = None

def get_conversation_manager(max_history: int = 10) -> ConversationManager:
    """
    Ottiene l'istanza singleton del ConversationManager
    
    Args:
        max_history: Numero massimo di messaggi in memoria
        
    Returns:
        ConversationManager instance
    """
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager(max_history)
    return _conversation_manager


if __name__ == "__main__":
    # Test
    manager = ConversationManager(max_history=5)
    
    session_id = manager.generate_session_id("test_user")
    print(f"Session ID: {session_id}")
    
    # Simula conversazione
    manager.add_message(session_id, "user", "Ciao!")
    manager.add_message(session_id, "assistant", "Ciao! Come posso aiutarti?")
    manager.add_message(session_id, "user", "Previsioni prodotto 40000")
    manager.add_message(session_id, "assistant", "Ecco le previsioni...")
    
    # Ottieni storico
    history = manager.get_conversation_history(session_id)
    print(f"\nStorico ({len(history)} messaggi):")
    for msg in history:
        print(f"  {msg['role']}: {msg['content']}")
    
    # Context summary
    print(f"\nContext Summary:\n{manager.get_context_summary(session_id)}")
