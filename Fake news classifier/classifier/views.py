import joblib
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
from .utils import TextPreprocessor

# Global variables for model components
model = None
vectorizer = None
preprocessor = None
model_loaded = False

def load_model():
    """Load ML model components on demand"""
    global model, vectorizer, preprocessor, model_loaded
    
    if model_loaded:
        return True
    
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        model = joblib.load(os.path.join(BASE_DIR, 'fake_news_model.joblib'))
        vectorizer = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.joblib'))
        
        # Create preprocessor instance directly
        preprocessor = TextPreprocessor()
        model_loaded = True
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def home(request):
    """Home page with the news classification form"""
    # Try to load model
    load_model()
    
    sample_texts = [
        "Scientists discover breakthrough treatment for Alzheimer's disease using new gene therapy approach",
        "Celebrity gossip: Famous actor caught in scandalous affair with politician's spouse",
        "Government announces new economic policy to address inflation and job growth",
        "Breaking: Aliens have landed and are secretly living among us, whistleblower claims"
    ]
    
    context = {
        'sample_texts': sample_texts,
        'model_loaded': model_loaded
    }
    return render(request, 'classifier/home.html', context)

@csrf_exempt
def predict_news(request):
    """API endpoint for news prediction"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            text = data.get('text', '').strip()
            
            if not text:
                return JsonResponse({'error': 'Please enter some text to analyze'}, status=400)
            
            if not load_model():
                return JsonResponse({'error': 'Model not loaded. Please check server configuration.'}, status=500)
            
            # Preprocess the text
            processed_text = preprocessor.preprocess_text(text)
            
            # Vectorize the text
            text_vector = vectorizer.transform([processed_text])
            
            # Make prediction
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]
            
            # Get confidence scores
            fake_confidence = float(probabilities[1]) * 100
            real_confidence = float(probabilities[0]) * 100
            
            # Determine result
            result = "FAKE" if prediction == 1 else "REAL"
            confidence = max(fake_confidence, real_confidence)
            
            response_data = {
                'prediction': result,
                'confidence': round(confidence, 2),
                'fake_confidence': round(fake_confidence, 2),
                'real_confidence': round(real_confidence, 2),
                'text_length': len(text),
                'processed_text': processed_text[:200] + "..." if len(processed_text) > 200 else processed_text
            }
            
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Prediction error: {str(e)}'}, status=500)
    
    return JsonResponse({'error': 'Only POST method allowed'}, status=405)

def about(request):
    """About page with information about the project"""
    context = {
        'model_accuracy': '89.12%',
        'training_samples': '20,800',
        'algorithm': 'Naive Bayes with TF-IDF',
        'features': 'Text preprocessing, N-gram analysis, Lemmatization'
    }
    return render(request, 'classifier/about.html', context)
