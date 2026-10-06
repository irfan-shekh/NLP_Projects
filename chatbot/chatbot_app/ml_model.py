import nltk
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
from . import nlp_utils

warnings.filterwarnings('ignore')

class ChatbotModel:
    def __init__(self):
        self.vectorizer = None
        self.corpus_vectors = None
        self.conversations = []
        self.is_trained = False
        
        # Download necessary NLTK data
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download necessary NLTK data if not already present"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
    
    def load_conversations(self, file_path):
        """Load conversations from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by lines and create conversation pairs
            lines = content.strip().split('\n')
            self.conversations = []
            
            for i in range(0, len(lines) - 1, 2):
                if i + 1 < len(lines):
                    user_input = lines[i].strip()
                    bot_response = lines[i + 1].strip()
                    if user_input and bot_response:
                        self.conversations.append({
                            'user': user_input,
                            'bot': bot_response
                        })
            
            return True
        except Exception as e:
            print(f"Error loading conversations: {e}")
            return False
    
    def preprocess_text(self, text):
        """Preprocess text using NLP utilities"""
        if not text.strip():
            return ""
        
        # Use the lemmatization function from nlp_utils
        processed = nlp_utils.lemmatization_sentence(text)
        return processed.lower()
    
    def train_model(self):
        """Train the chatbot model using TF-IDF vectorization"""
        if not self.conversations:
            return False
        
        # Extract user inputs for training
        user_inputs = [conv['user'] for conv in self.conversations]
        
        # Preprocess all user inputs
        processed_inputs = [self.preprocess_text(text) for text in user_inputs]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform the corpus
        self.corpus_vectors = self.vectorizer.fit_transform(processed_inputs)
        
        # Save the model
        self.save_model()
        
        self.is_trained = True
        return True
    
    def get_response(self, user_input):
        """Get response for user input"""
        if not self.is_trained or not self.vectorizer:
            return "I'm still learning. Please try again later."
        
        # Preprocess user input
        processed_input = self.preprocess_text(user_input)
        
        if not processed_input:
            return "I didn't understand that. Could you please rephrase?"
        
        # Transform user input
        user_vector = self.vectorizer.transform([processed_input])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector, self.corpus_vectors)
        
        # Get the best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[0, best_match_idx]
        
        # Threshold for similarity
        if best_similarity > 0.2:  # Adjust threshold as needed
            return self.conversations[best_match_idx]['bot']
        else:
            # Default responses for low similarity
            default_responses = [
                "I'm not sure how to respond to that. Could you try asking differently?",
                "That's interesting! I don't have a specific response for that.",
                "I'm still learning about that topic. Could you ask something else?",
                "I don't have enough information to answer that. Can you help me learn?"
            ]
            return np.random.choice(default_responses)
    
    def save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'corpus_vectors': self.corpus_vectors,
                'conversations': self.conversations,
                'is_trained': self.is_trained
            }
            
            with open('chatbot_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            with open('chatbot_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data['vectorizer']
            self.corpus_vectors = model_data['corpus_vectors']
            self.conversations = model_data['conversations']
            self.is_trained = model_data['is_trained']
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Global model instance
chatbot_model = ChatbotModel()
