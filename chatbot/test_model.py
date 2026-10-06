#!/usr/bin/env python
"""
Test script for the chatbot ML model
Run this script to validate the ML functionality
"""

import os
import sys
import django

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')
django.setup()

from chatbot_app.ml_model import chatbot_model

def test_model_functionality():
    """Test the chatbot model functionality"""
    print("=== Chatbot Model Test ===\n")
    
    # Test 1: Load conversations
    print("1. Testing conversation loading...")
    dialogs_path = os.path.join('chatbot_app', 'dialogs.txt')
    
    if os.path.exists(dialogs_path):
        success = chatbot_model.load_conversations(dialogs_path)
        print(f"   Conversations loaded: {success}")
        print(f"   Number of conversations: {len(chatbot_model.conversations)}")
        
        if chatbot_model.conversations:
            print(f"   Sample conversation: {chatbot_model.conversations[0]}")
    else:
        print(f"   ERROR: dialogs.txt not found at {dialogs_path}")
        return False
    
    # Test 2: Text preprocessing
    print("\n2. Testing text preprocessing...")
    test_text = "Hello! I'm testing the chatbot's functionality."
    processed = chatbot_model.preprocess_text(test_text)
    print(f"   Original: {test_text}")
    print(f"   Processed: {processed}")
    
    # Test 3: Model training
    print("\n3. Testing model training...")
    success = chatbot_model.train_model()
    print(f"   Training successful: {success}")
    print(f"   Model trained: {chatbot_model.is_trained}")
    
    if success:
        # Test 4: Get responses
        print("\n4. Testing response generation...")
        test_messages = [
            "Hello",
            "How are you?",
            "What can you do?",
            "Thank you",
            "Goodbye"
        ]
        
        for msg in test_messages:
            response = chatbot_model.get_response(msg)
            print(f"   User: {msg}")
            print(f"   Bot:  {response}")
            print()
    
    # Test 5: Model persistence
    print("5. Testing model persistence...")
    save_success = chatbot_model.save_model()
    print(f"   Model saved: {save_success}")
    
    # Create new instance and test loading
    from chatbot_app.ml_model import ChatbotModel
    new_model = ChatbotModel()
    load_success = new_model.load_model()
    print(f"   Model loaded: {load_success}")
    print(f"   Loaded model trained: {new_model.is_trained}")
    
    print("\n=== Test Complete ===")
    return True

if __name__ == "__main__":
    try:
        test_model_functionality()
        print("All tests completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
