from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
import os
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from .forms import CommentForm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Preprocessing function
def preprocess_text(text):
    """Preprocess text to match training data format"""
    # Remove alphanumeric characters
    alphanumeric = lambda x: re.sub(r'\w*\d\w*', ' ', x)
    # Remove punctuation and convert to lowercase
    punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
    # Remove newlines
    remove_n = lambda x: re.sub(r"\n", " ", x)
    # Remove non-ASCII characters
    remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]', r' ', x)
    
    text = alphanumeric(text)
    text = punc_lower(text)
    text = remove_n(text)
    text = remove_non_ascii(text)
    
    return text

# Load model and vectorizer
def load_model():
    """Load the trained model and vectorizer"""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'toxic_model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.pkl')
    
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# Home page
def home(request):
    """Render the home page with comment form"""
    form = CommentForm()
    result = None
    
    if request.method == 'POST':
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.cleaned_data['comment']
            
            # Load model and vectorizer
            model, vectorizer = load_model()
            
            if model is None or vectorizer is None:
                result = {
                    'error': 'Model not found. Please train the model first.',
                    'is_toxic': False,
                    'confidence': 0
                }
            else:
                # Preprocess the comment
                processed_comment = preprocess_text(comment)
                
                # Vectorize the comment
                comment_vector = vectorizer.transform([processed_comment])
                
                # Get prediction and probability
                prediction = model.predict(comment_vector)[0]
                probability = model.predict_proba(comment_vector)[0][1]
                
                result = {
                    'is_toxic': bool(prediction),
                    'confidence': round(probability * 100, 2),
                    'original_comment': comment,
                    'processed_comment': processed_comment
                }
    
    context = {
        'form': form,
        'result': result
    }
    
    return render(request, 'classifier/home.html', context)

# API endpoint for predictions
@csrf_exempt
def predict_api(request):
    """API endpoint for programmatic access"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
    comment = request.POST.get('comment', '')
    if not comment:
        return JsonResponse({'error': 'No comment provided'}, status=400)
    
    # Load model and vectorizer
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        return JsonResponse({'error': 'Model not found. Please train the model first.'}, status=500)
    
    # Preprocess the comment
    processed_comment = preprocess_text(comment)
    
    # Vectorize the comment
    comment_vector = vectorizer.transform([processed_comment])
    
    # Get prediction and probability
    prediction = model.predict(comment_vector)[0]
    probability = model.predict_proba(comment_vector)[0][1]
    
    return JsonResponse({
        'is_toxic': bool(prediction),
        'confidence': round(probability * 100, 2),
        'comment': comment
    })
