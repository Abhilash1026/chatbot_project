# app.py
import streamlit as st
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os
from typing import List, Tuple
import base64

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set Streamlit page config with modern theme
st.set_page_config(
    page_title="Jarvis AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    :root {
        --primary-color: #4f46e5;
        --secondary-color: #f9fafb;
        --accent-color: #10b981;
        --text-color: #111827;
        --light-text: #6b7280;
    }
    
    .stTextInput input {
        border-radius: 20px;
        padding: 12px 16px;
        border: 1px solid #d1d5db;
    }
    
    .stButton button {
        border-radius: 20px;
        background-color: var(--primary-color);
        color: white;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #4338ca;
        transform: translateY(-2px);
    }
    
    .chat-message {
        padding: 14px 18px;
        border-radius: 20px;
        margin-bottom: 12px;
        display: inline-block;
        max-width: 75%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        line-height: 1.5;
    }
    
    .user-message {
        background-color: var(--primary-color);
        color: white;
        margin-left: 25%;
        text-align: right;
        border-bottom-right-radius: 5px;
    }
    
    .bot-message {
        background-color: var(--secondary-color);
        color: var(--text-color);
        margin-right: 25%;
        border-bottom-left-radius: 5px;
    }
    
    .sidebar .sidebar-content {
        background-color: var(--secondary-color);
        padding: 1.5rem;
    }
    
    .stRadio div[role="radiogroup"] {
        gap: 0.5rem;
    }
    
    .stRadio div[role="radio"] {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }
    
    .stRadio div[role="radio"]:hover {
        background-color: #f3f4f6;
    }
    
    .stRadio div[role="radio"][aria-checked="true"] {
        background-color: var(--primary-color);
        color: white;
        border-color: var(--primary-color);
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-color);
    }
    
    .stSuccess {
        background-color: #ecfdf5;
        color: #065f46;
    }
    
    .stWarning {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .stError {
        background-color: #fee2e2;
        color: #b91c1c;
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        color: var(--light-text);
        font-size: 0.9rem;
    }
    
    .history-item {
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 12px;
        background-color: white;
        border: 1px solid #e5e7eb;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .history-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .delete-btn {
        background-color: #fee2e2 !important;
        color: #b91c1c !important;
        border: 1px solid #fca5a5 !important;
    }
    
    .delete-btn:hover {
        background-color: #fecaca !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced conversation data
corpus = """
Hello! Hi there! Hey! Greetings! Welcome!
What can you do? I can answer questions, provide information, and have conversations.
How are you? I'm doing great, thanks for asking! I'm just a chatbot, but I'm always happy to help.
What's your name? I'm an AI chatbot created to assist you. You can call me Jarvis.
Who created you? I was developed by a team of AI enthusiasts using Python and NLP techniques.
What time is it? The current time is {current_time}.
What day is today? Today is {current_day}.
Goodbye! Bye! See you later! Take care! Farewell! Goodbye! It was nice chatting with you. Hope to see you again soon!
Thanks! Thank you! I appreciate it! You're welcome! Happy to help! No problem at all!
What is NLP? NLP stands for Natural Language Processing. It's a field of AI that focuses on interactions between computers and human language.
Tell me a joke. Why don't scientists trust atoms? Because they make up everything!
What's the weather like? I don't have real-time weather data, but you can check your favorite weather app for updates.
How does this work? I use NLP techniques like tokenization and cosine similarity to understand and respond to your messages.
What can you help me with? I can assist with general knowledge, answer questions, tell jokes, and more!
"""

# Preprocessing functions
def preprocess_text(text: str) -> str:
    """Clean and preprocess text for NLP processing."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation (basic)
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    text = ' '.join([word for word in word_tokens if word not in stop_words])
    return text

# Initialize vectorizer and similarity matrix
def initialize_nlp_components():
    """Initialize NLP components and compute similarity matrix."""
    global sent_tokens, tfidf_vectorizer, tfidf_matrix
    
    # Tokenize sentences
    sent_tokens = nltk.sent_tokenize(corpus)
    
    # Preprocess all sentences
    processed_sentences = [preprocess_text(sent) for sent in sent_tokens]
    
    # Initialize and fit TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_sentences)

# Initialize NLP components
initialize_nlp_components()

# Chatbot response generation
def generate_response(user_input: str) -> str:
    """Generate response to user input using TF-IDF and cosine similarity."""
    try:
        # Preprocess user input
        processed_input = preprocess_text(user_input)
        
        # Vectorize user input
        input_vector = tfidf_vectorizer.transform([processed_input])
        
        # Compute similarity scores
        similarity_scores = cosine_similarity(input_vector, tfidf_matrix)
        
        # Get index of most similar sentence
        best_match_idx = np.argmax(similarity_scores)
        
        # Check if similarity score is above threshold
        if similarity_scores[0][best_match_idx] > 0.3:
            return sent_tokens[best_match_idx]
        else:
            return "I'm not sure I understand. Could you rephrase that or ask something else?"
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I encountered an error processing your request. Please try again."

# Greeting responses
def greet(user_input: str) -> str:
    """Check for greetings in user input."""
    greetings = ['hello', 'hi', 'hey', 'greetings', 'welcome']
    farewells = ['bye', 'goodbye', 'see you', 'farewell']
    thanks = ['thanks', 'thank you', 'appreciate it']
    
    user_input = user_input.lower()
    
    if any(word in user_input for word in greetings):
        return np.random.choice([
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! Nice to see you. What's on your mind?"
        ])
    elif any(word in user_input for word in farewells):
        return np.random.choice([
            "Goodbye! Have a wonderful day!",
            "See you later! Come back anytime.",
            "Farewell! It was nice chatting with you."
        ])
    elif any(word in user_input for word in thanks):
        return np.random.choice([
            "You're welcome! 😊",
            "Happy to help!",
            "No problem at all! Let me know if you need anything else."
        ])
    return None

def save_conversation_history(conversation: List[Tuple[str, str]]) -> str:
    """Save conversation history to a CSV file with enhanced features."""
    try:
        if not conversation:
            st.warning("No conversation to save.")
            return None
            
        # Create a DataFrame with additional metadata
        df = pd.DataFrame(conversation, columns=['Speaker', 'Message'])
        df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create directory if it doesn't exist
        os.makedirs("chat_history", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history/conversation_{timestamp}.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        # Create a download link
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Conversation</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        st.success("Conversation saved successfully!")
        return filename
    except Exception as e:
        st.error(f"Error saving conversation: {e}")
        return None

def load_conversation_history() -> pd.DataFrame:
    """Load all conversation history with enhanced features."""
    try:
        if not os.path.exists("chat_history"):
            return None
            
        history_files = [f for f in os.listdir("chat_history") if f.startswith('conversation_') and f.endswith('.csv')]
        history_data = []

        for file in history_files:
            try:
                df = pd.read_csv(f"chat_history/{file}")
                # Extract timestamp from filename
                timestamp_str = file.replace("conversation_", "").replace(".csv", "")
                readable_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                df['File'] = file
                df['ReadableDate'] = readable_time
                history_data.append(df)
            except Exception as e:
                st.error(f"Error loading {file}: {e}")

        return pd.concat(history_data, ignore_index=True) if history_data else None
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return None

def delete_conversation_file(filename: str) -> bool:
    """Delete a specific conversation file."""
    try:
        if os.path.exists(f"chat_history/{filename}"):
            os.remove(f"chat_history/{filename}")
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting file: {e}")
        return False

def delete_all_conversations() -> bool:
    """Delete all conversation history."""
    try:
        if os.path.exists("chat_history"):
            for file in os.listdir("chat_history"):
                if file.startswith('conversation_') and file.endswith('.csv'):
                    os.remove(f"chat_history/{file}")
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting conversations: {e}")
        return False

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'selected_history' not in st.session_state:
    st.session_state.selected_history = None

# Sidebar menu with enhanced options
with st.sidebar:
    st.title("🤖 Jarvis Chatbot")
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        ["💬 Chat", "📜 History", "⚙️ Settings"],
        index=0
    )
    
    st.markdown("---")
    
    if st.button("🧹 Clear Current Chat"):
        st.session_state.conversation = []
        st.rerun()
    
    if st.session_state.conversation:
        if st.button("💾 Save Conversation"):
            filename = save_conversation_history(st.session_state.conversation)
    
    st.markdown("---")
    st.markdown("### Quick Actions")
    
    if menu == "📜 History":
        if st.button("🗑️ Delete All History", key="delete_all_sidebar"):
            if delete_all_conversations():
                st.success("All conversation history deleted!")
                st.rerun()
            else:
                st.error("Failed to delete history")

# Main app content
if menu == "💬 Chat":
    st.title("💬 Chat with Jarvis")
    
    # User info section
    if not st.session_state.user_name:
        with st.expander("👤 Tell me about yourself", expanded=True):
            name = st.text_input("What's your name?")
            if st.button("Save Name"):
                if name:
                    st.session_state.user_name = name
                    st.session_state.conversation.append(
                        ("System", f"User provided their name: {name}")
                    )
                    st.rerun()
                else:
                    st.warning("Please enter a name")
    
    # Display conversation
    chat_container = st.container()
    
    with chat_container:
        for speaker, message in st.session_state.conversation:
            if speaker == "System":
                continue
                
            if speaker == "You":
                st.markdown(
                    f'<div class="chat-message user-message"><b>You</b>: {message}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message bot-message"><b>Jarvis</b>: {message}</div>',
                    unsafe_allow_html=True
                )
    
    # User input with enhanced features
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Type your message:",
                placeholder="Ask me anything...",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("🚀 Send")
        
        if submitted and user_input:
            # Add user message to conversation
            st.session_state.conversation.append(("You", user_input))
            
            # Generate response
            with st.spinner("Jarvis is thinking..."):
                time.sleep(0.3)  # Simulate processing time
                
                user_input = user_input.lower()
                if user_input == 'bye':
                    bot_reply = "Goodbye! Take care ❤️"
                elif greet(user_input) is not None:
                    bot_reply = greet(user_input)
                else:
                    # Replace dynamic content
                    if "{current_time}" in corpus or "{current_day}" in corpus:
                        current_time = datetime.now().strftime("%H:%M")
                        current_day = datetime.now().strftime("%A, %B %d, %Y")
                        modified_corpus = corpus.replace("{current_time}", current_time)
                        modified_corpus = modified_corpus.replace("{current_day}", current_day)
                        global sent_tokens
                        sent_tokens = nltk.sent_tokenize(modified_corpus)
                        initialize_nlp_components()
                    
                    bot_reply = generate_response(user_input)
                
                st.session_state.conversation.append(("Chatbot", bot_reply))
                st.rerun()

elif menu == "📜 History":
    st.title("📜 Conversation History")
    
    history_data = load_conversation_history()
    
    if history_data is not None and not history_data.empty:
        # Show statistics
        st.subheader("📊 Chat Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Conversations", len(history_data['File'].unique()))
        with col2:
            st.metric("Total Messages", len(history_data))
        with col3:
            st.metric("Your Messages", sum(history_data['Speaker'] == 'You'))
        
        # History management section
        st.subheader("🗃️ History Management")
        
        # Delete all button
        if st.button("🗑️ Delete All History", key="delete_all_main", type="secondary"):
            if delete_all_conversations():
                st.success("All conversation history deleted!")
                st.rerun()
            else:
                st.error("Failed to delete history")
        
        # Filter options
        st.subheader("🔍 Filter Conversations")
        selected_file = st.selectbox(
            "Select conversation",
            options=sorted(history_data['File'].unique(), reverse=True),
            format_func=lambda x: datetime.strptime(
                x.replace("conversation_", "").replace(".csv", ""), 
                "%Y%m%d_%H%M%S"
            ).strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Delete specific conversation
        if st.button(f"🗑️ Delete Selected", key="delete_selected"):
            if delete_conversation_file(selected_file):
                st.success("Conversation deleted!")
                st.rerun()
            else:
                st.error("Failed to delete conversation")
        
        # Display filtered conversation
        filtered_data = history_data[history_data['File'] == selected_file]
        
        st.markdown("---")
        st.subheader(f"🗓 Conversation from {filtered_data['ReadableDate'].iloc[0]}")
        
        for _, row in filtered_data.iterrows():
            if row['Speaker'] == "You":
                st.markdown(
                    f'<div class="chat-message user-message"><b>You</b>: {row["Message"]}</div>',
                    unsafe_allow_html=True
                )
            elif row['Speaker'] == "Chatbot":
                st.markdown(
                    f'<div class="chat-message bot-message"><b>Jarvis</b>: {row["Message"]}</div>',
                    unsafe_allow_html=True
                )
            st.markdown("---")
        
        # Download button for specific conversation
        csv = filtered_data[['Speaker', 'Message', 'Timestamp']].to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_{selected_file}">💾 Download This Conversation</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("No conversation history found. Start chatting to create some history!")

elif menu == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    st.subheader("👤 Personalization")
    new_name = st.text_input(
        "Change your name",
        value=st.session_state.user_name if st.session_state.user_name else ""
    )
    
    if st.button("Save Name"):
        if new_name and new_name != st.session_state.user_name:
            st.session_state.user_name = new_name
            st.success("Name updated successfully!")
        else:
            st.warning("Please enter a new name")
    
    st.subheader("🤖 Chatbot Behavior")
    response_speed = st.slider(
        "Response speed (simulated)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Adjust the simulated response time of the chatbot"
    )
    
    st.subheader("📊 Data Management")
    if st.button("🗑️ Delete All Conversation History", type="secondary"):
        if delete_all_conversations():
            st.success("All conversation history has been deleted.")
            st.rerun()
        else:
            st.error("Failed to delete history")

# Footer
st.markdown("---")
st.markdown(
    """
    <div class="footer">
    <p>Developed By Abhilash | Jarvis Chatbot v2.0</p>
    </div>
    """,
    unsafe_allow_html=True
)