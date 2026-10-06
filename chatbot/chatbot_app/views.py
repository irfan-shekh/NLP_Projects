from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import os
from .ml_model import chatbot_model

def home(request):
    """Home page with chatbot interface"""
    return render(request, 'chatbot_app/home.html')

@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    """Handle chat requests and return bot response"""
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return JsonResponse({
                'error': 'Message cannot be empty'
            }, status=400)
        
        # Get response from chatbot model
        bot_response = chatbot_model.get_response(user_message)
        
        return JsonResponse({
            'response': bot_response,
            'status': 'success'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON format'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': f'An error occurred: {str(e)}'
        }, status=500)

def train_model(request):
    """Train or retrain the chatbot model"""
    try:
        # Load conversations from dialogs.txt
        dialogs_path = os.path.join(os.path.dirname(__file__), 'dialogs.txt')
        
        if not os.path.exists(dialogs_path):
            return JsonResponse({
                'error': 'Training data file not found'
            }, status=404)
        
        # Load conversations
        success = chatbot_model.load_conversations(dialogs_path)
        
        if not success:
            return JsonResponse({
                'error': 'Failed to load training data'
            }, status=500)
        
        # Train the model
        success = chatbot_model.train_model()
        
        if success:
            return JsonResponse({
                'message': 'Model trained successfully',
                'conversations_loaded': len(chatbot_model.conversations),
                'status': 'success'
            })
        else:
            return JsonResponse({
                'error': 'Failed to train model'
            }, status=500)
            
    except Exception as e:
        return JsonResponse({
            'error': f'Training error: {str(e)}'
        }, status=500)

def model_status(request):
    """Get current model status"""
    try:
        # Try to load existing model
        if not chatbot_model.is_trained:
            chatbot_model.load_model()
        
        return JsonResponse({
            'is_trained': chatbot_model.is_trained,
            'conversations_count': len(chatbot_model.conversations),
            'status': 'success'
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'Status check error: {str(e)}'
        }, status=500)
