# conversation_history.py
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import streamlit as st

class ConversationHistory:
    """
    A class to manage conversation history with enhanced features including:
    - Persistent storage to JSON/CSV
    - Search capabilities
    - Statistics generation
    - Session management
    """
    
    def __init__(self, storage_file: str = "conversation_history.json"):
        """
        Initialize the conversation history manager.
        
        Args:
            storage_file: Path to file where history will be stored
        """
        self.storage_file = storage_file
        self.conversation_log: List[Dict] = []
        self._load_history()
    
    def _load_history(self) -> None:
        """Load conversation history from storage file if it exists."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    self.conversation_log = json.load(f)
        except Exception as e:
            st.error(f"Error loading conversation history: {e}")
            self.conversation_log = []
    
    def _save_history(self) -> None:
        """Save current conversation history to storage file."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.conversation_log, f, indent=2)
        except Exception as e:
            st.error(f"Error saving conversation history: {e}")
    
    def add_to_history(self, user_input: str, bot_response: str, 
                      metadata: Optional[Dict] = None) -> None:
        """
        Add a new conversation exchange to the history.
        
        Args:
            user_input: The user's message
            bot_response: The bot's response
            metadata: Additional context data (timestamp, sentiment, etc.)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "bot": bot_response,
            "metadata": metadata or {}
        }
        
        self.conversation_log.append(entry)
        self._save_history()
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get the conversation history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of conversation entries
        """
        if limit is not None:
            return self.conversation_log[-limit:]
        return self.conversation_log
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Return conversation history as a pandas DataFrame."""
        return pd.DataFrame(self.conversation_log)
    
    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.conversation_log = []
        self._save_history()
    
    def export_to_csv(self, file_path: str = "conversation_history.csv") -> bool:
        """
        Export conversation history to CSV file.
        
        Args:
            file_path: Path to save the CSV file
            
        Returns:
            True if export was successful
        """
        try:
            df = self.get_history_dataframe()
            df.to_csv(file_path, index=False)
            return True
        except Exception as e:
            st.error(f"Error exporting to CSV: {e}")
            return False
    
    def search_history(self, query: str, search_in: str = "both") -> List[Dict]:
        """
        Search conversation history for specific text.
        
        Args:
            query: Text to search for
            search_in: Where to search ('user', 'bot', or 'both')
            
        Returns:
            List of matching conversation entries
        """
        results = []
        query = query.lower()
        
        for entry in self.conversation_log:
            user_match = search_in in ["user", "both"] and query in entry["user"].lower()
            bot_match = search_in in ["bot", "both"] and query in entry["bot"].lower()
            
            if user_match or bot_match:
                results.append(entry)
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        Generate statistics about the conversation history.
        
        Returns:
            Dictionary containing various statistics
        """
        if not self.conversation_log:
            return {}
            
        df = self.get_history_dataframe()
        
        return {
            "total_conversations": len(df),
            "user_message_count": sum(1 for _ in df['user']),
            "bot_message_count": sum(1 for _ in df['bot']),
            "first_conversation": df['timestamp'].min(),
            "last_conversation": df['timestamp'].max(),
            "avg_user_message_length": df['user'].str.len().mean(),
            "avg_bot_message_length": df['bot'].str.len().mean()
        }
    
    def get_conversation_session(self, session_id: str) -> List[Dict]:
        """
        Get conversation history for a specific session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            List of conversation entries for the session
        """
        return [entry for entry in self.conversation_log 
                if entry.get("metadata", {}).get("session_id") == session_id]

# Singleton instance for easy access
conversation_history = ConversationHistory()

# Helper functions for backward compatibility
def add_to_history(user_input: str, bot_response: str) -> None:
    """Legacy function to add to history."""
    conversation_history.add_to_history(user_input, bot_response)

def get_history() -> List[Dict]:
    """Legacy function to get full history."""
    return conversation_history.get_history()