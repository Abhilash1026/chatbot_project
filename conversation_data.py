# conversation_data.py
from typing import Dict, List
import json
import os
from datetime import datetime

class KnowledgeBase:
    """
    A comprehensive knowledge management system for the chatbot with:
    - Multiple domain support
    - Dynamic content
    - Contextual responses
    - Personalization
    - Persistent storage
    """
    
    def __init__(self, initial_corpus: str = ""):
        """
        Initialize the knowledge base with optional initial corpus.
        
        Args:
            initial_corpus: Initial text corpus for the knowledge base
        """
        self.corpus = initial_corpus
        self.domain_knowledge: Dict[str, List[str]] = {
            "greetings": [
                "Hello! I'm your assistant. How can I help you today?",
                "Hi there! What can I do for you?",
                "Greetings! I'm here to assist you."
            ],
            "credit": [
                "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame.",
                "Improving your credit score involves paying bills on time and keeping credit utilization low.",
                "Credit scores typically range from 300 to 850, with higher being better."
            ],
            "geography": [
                "The capital of France is Paris.",
                "Mount Everest is the highest mountain in the world.",
                "The Nile is the longest river on Earth."
            ],
            "weather": [
                "The weather today is sunny and bright.",
                "You can check current weather conditions using weather apps or websites.",
                "Weather forecasts predict rain later this week."
            ],
            "personal_finance": [
                "Creating a budget is the first step to financial health.",
                "An emergency fund should cover 3-6 months of expenses.",
                "Investing early can significantly grow your wealth over time."
            ]
        }
        self.user_specific_data: Dict[str, Dict] = {}
        self.storage_file = "knowledge_base.json"
        self._load_knowledge()
    
    def _load_knowledge(self) -> None:
        """Load knowledge from persistent storage if available."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    self.domain_knowledge.update(data.get("domain_knowledge", {}))
                    self.user_specific_data.update(data.get("user_specific_data", {}))
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
    
    def _save_knowledge(self) -> None:
        """Save current knowledge to persistent storage."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump({
                    "domain_knowledge": self.domain_knowledge,
                    "user_specific_data": self.user_specific_data
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
    
    def get_corpus(self) -> str:
        """
        Get the complete knowledge corpus with dynamic content.
        
        Returns:
            The complete knowledge corpus as a single string
        """
        dynamic_corpus = self.corpus.replace("{time}", datetime.now().strftime("%H:%M"))
        dynamic_corpus = dynamic_corpus.replace("{date}", datetime.now().strftime("%A, %B %d, %Y"))
        
        # Add all domain knowledge to the corpus
        for domain, facts in self.domain_knowledge.items():
            dynamic_corpus += "\n".join(facts) + "\n"
        
        return dynamic_corpus
    
    def add_domain_knowledge(self, domain: str, facts: List[str]) -> None:
        """
        Add new knowledge to a specific domain.
        
        Args:
            domain: The knowledge domain (e.g., 'credit', 'weather')
            facts: List of facts to add to the domain
        """
        if domain not in self.domain_knowledge:
            self.domain_knowledge[domain] = []
        self.domain_knowledge[domain].extend(facts)
        self._save_knowledge()
    
    def add_user_specific_data(self, user_id: str, data: Dict) -> None:
        """
        Add user-specific information to the knowledge base.
        
        Args:
            user_id: Unique identifier for the user
            data: Dictionary of user-specific information
        """
        if user_id not in self.user_specific_data:
            self.user_specific_data[user_id] = {}
        self.user_specific_data[user_id].update(data)
        self._save_knowledge()
    
    def get_user_data(self, user_id: str) -> Dict:
        """
        Get all stored information about a specific user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary of user-specific information
        """
        return self.user_specific_data.get(user_id, {})
    
    def search_knowledge(self, query: str, domain: str = None) -> List[str]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search term
            domain: Optional specific domain to search
            
        Returns:
            List of matching knowledge entries
        """
        results = []
        query = query.lower()
        
        search_domains = [domain] if domain else self.domain_knowledge.keys()
        
        for domain in search_domains:
            for fact in self.domain_knowledge.get(domain, []):
                if query in fact.lower():
                    results.append(fact)
        
        return results
    
    def get_random_fact(self, domain: str = None) -> str:
        """
        Get a random fact from the knowledge base.
        
        Args:
            domain: Optional specific domain to choose from
            
        Returns:
            A random fact as a string
        """
        if domain:
            return random.choice(self.domain_knowledge.get(domain, [""]))
        
        all_facts = []
        for facts in self.domain_knowledge.values():
            all_facts.extend(facts)
        return random.choice(all_facts) if all_facts else ""

# Initialize with default corpus
default_corpus = """
Hello! I'm your assistant. You can ask me questions.
You can check your credit score for free on several websites such as Credit Karma and Credit Sesame.
The weather today is sunny and bright.
The capital of France is Paris.
You can improve your credit score by paying your bills on time and reducing credit card debt.
Have a great day!
Current time is {time}.
Today's date is {date}.
"""

# Singleton instance for easy access
knowledge_base = KnowledgeBase(default_corpus)

# Legacy variable for backward compatibility
corpus = knowledge_base.get_corpus()