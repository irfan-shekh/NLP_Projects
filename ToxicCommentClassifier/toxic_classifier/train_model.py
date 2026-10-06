"""
Train and save the toxic comment classification model.
This script trains a Random Forest classifier on the toxic comment dataset
and saves the model and vectorizer for use in the Django application.
"""

import pandas as pd
import numpy as np
import joblib
import os
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.join(BASE_DIR, '..', 'Notebook')
DATA_PATH = os.path.join(NOTEBOOK_DIR, 'train.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'toxic_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

def preprocess_text(text):
    """Preprocess text to match training data format"""
    alphanumeric = lambda x: re.sub(r'\w*\d\w*', ' ', x)
    punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
    remove_n = lambda x: re.sub(r"\n", " ", x)
    remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]', r' ', x)
    
    text = alphanumeric(text)
    text = punc_lower(text)
    text = remove_n(text)
    text = remove_non_ascii(text)
    
    return text

def train_model():
    """Train and save the toxic comment classification model"""
    print("Loading dataset...")
    
    # Load the dataset
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}")
        print("Please ensure train.csv exists in the Notebook directory.")
        return False
    
    print(f"Dataset loaded with {len(df)} rows")
    
    # Preprocess the comment text
    print("Preprocessing text...")
    df['comment_text'] = df['comment_text'].apply(preprocess_text)
    
    # Create balanced dataset for toxic comments
    print("Creating balanced dataset...")
    toxic_1 = df[df['toxic'] == 1].iloc[:5000]
    toxic_0 = df[df['toxic'] == 0].iloc[:5000]
    balanced_df = pd.concat([toxic_1, toxic_0])
    
    print(f"Balanced dataset created with {len(balanced_df)} rows")
    
    # Split the data
    X = balanced_df.comment_text
    y = balanced_df['toxic']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create and fit TF-IDF vectorizer
    print("Creating TF-IDF vectorizer...")
    tfv = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    X_train_fit = tfv.fit_transform(X_train)
    X_test_fit = tfv.transform(X_test)
    
    # Train Random Forest classifier
    print("Training Random Forest classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=50)
    rf.fit(X_train_fit, y_train)
    
    # Evaluate the model
    train_score = rf.score(X_train_fit, y_train)
    test_score = rf.score(X_test_fit, y_test)
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save the model and vectorizer
    print(f"Saving model to {MODEL_PATH}")
    print(f"Saving vectorizer to {VECTORIZER_PATH}")
    
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(tfv, VECTORIZER_PATH)
    
    print("Model and vectorizer saved successfully!")
    
    # Test the model with sample comments
    print("\nTesting model with sample comments:")
    test_comments = [
        'i killed an insect and ate it',
        'Is this sentence a good one',
        'truth will prevail'
    ]
    
    for comment in test_comments:
        processed = preprocess_text(comment)
        vectorized = tfv.transform([processed])
        prob = rf.predict_proba(vectorized)[0][1]
        pred = rf.predict(vectorized)[0]
        print(f"'{comment}' -> Toxic: {bool(pred)}, Confidence: {prob:.2%}")
    
    return True

if __name__ == '__main__':
    success = train_model()
    if success:
        print("\nModel training completed successfully!")
    else:
        print("\nModel training failed!")
