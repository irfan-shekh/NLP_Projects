import pandas as pd
import numpy as np
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Contraction mapping
CONTRACTION_MAP = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
    "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is",
    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
    "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
    "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
    "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
    "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
    "so's": "so as", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
    "there'd": "there would", "there'd've": "there would have", "there's": "there is", "they'd": "they would",
    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are",
    "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
    "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
    "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
    "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
    "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
    "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
    "you'll've": "you will have", "you're": "you are", "you've": "you have"
}

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def contraction_remove(self, text):
        for key, value in CONTRACTION_MAP.items():
            text = re.sub(r"{}".format(re.escape(key)), '{}'.format(value), text, flags=re.IGNORECASE)
        return text
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove contractions
        text = self.contraction_remove(text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token, pos_tag_token in pos_tag(tokens):
            if token not in self.stop_words and len(token) > 2:
                # Determine POS for lemmatization
                if pos_tag_token.startswith('V'):
                    pos_val = 'v'
                elif pos_tag_token.startswith('J'):
                    pos_val = 'a'
                elif pos_tag_token.startswith('R'):
                    pos_val = 'r'
                else:
                    pos_val = 'n'
                
                lemmatized_token = self.lemmatizer.lemmatize(token, pos_val)
                processed_tokens.append(lemmatized_token)
        
        return ' '.join(processed_tokens)

def main():
    print("Loading and preprocessing data...")
    
    # Load the dataset
    df = pd.read_csv('Notebook/train.csv')
    
    # Drop null values
    df = df.dropna()
    
    # Combine title and text for better features
    df['combined_text'] = df['title'] + ' ' + df['text']
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess the text
    print("Preprocessing text data...")
    df['processed_text'] = df['combined_text'].apply(preprocessor.preprocess_text)
    
    # Prepare features and target
    X = df['processed_text']
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create TF-IDF vectorizer
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.8)
    
    # Fit and transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Naive Bayes model
    print("Training Naive Bayes model...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Save the model and vectorizer
    joblib.dump(model, 'fake_news_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(preprocessor, 'text_preprocessor.joblib')
    
    print("Model and preprocessing components saved successfully!")
    
    # Test with sample texts
    sample_texts = [
        "Breaking: Scientists discover cure for cancer in major breakthrough",
        "Aliens landed in Times Square yesterday, eyewitnesses claim",
        "President signs new healthcare bill into law",
        "Celebrity caught in scandalous affair with politician"
    ]
    
    print("\nTesting model with sample texts:")
    for i, text in enumerate(sample_texts):
        processed = preprocessor.preprocess_text(text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]
        
        result = "FAKE" if prediction == 1 else "REAL"
        confidence = max(probability) * 100
        print(f"{i+1}. '{text[:50]}...' -> {result} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    main()
