#!/usr/bin/env python
"""
Setup script for the AI Chatbot application
Run this script to set up the environment and train the initial model
"""

import os
import sys
import subprocess
import django

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def setup_project():
    """Set up the complete project"""
    print("=== AI Chatbot Setup ===")
    
    # Check if virtual environment is active
    if sys.prefix == sys.base_prefix:
        print("⚠ Warning: No virtual environment detected")
        print("  It's recommended to use a virtual environment")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Run Django migrations
    if not run_command("python manage.py migrate", "Running database migrations"):
        return False
    
    # Check Django setup
    if not run_command("python manage.py check", "Checking Django configuration"):
        return False
    
    # Set up Django for model training
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chatbot_project.settings')
    django.setup()
    
    # Train the model
    print("\nTraining the chatbot model...")
    try:
        from chatbot_app.ml_model import chatbot_model
        
        # Load conversations
        dialogs_path = os.path.join('chatbot_app', 'dialogs.txt')
        if os.path.exists(dialogs_path):
            success = chatbot_model.load_conversations(dialogs_path)
            if success:
                print(f"✓ Loaded {len(chatbot_model.conversations)} conversations")
            else:
                print("✗ Failed to load conversations")
                return False
        else:
            print(f"✗ Training data not found at {dialogs_path}")
            return False
        
        # Train model
        success = chatbot_model.train_model()
        if success:
            print("✓ Model trained successfully")
        else:
            print("✗ Model training failed")
            return False
        
        # Save model
        success = chatbot_model.save_model()
        if success:
            print("✓ Model saved successfully")
        else:
            print("✗ Failed to save model")
            return False
            
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return False
    
    # Collect static files
    if not run_command("python manage.py collectstatic --noinput", "Collecting static files"):
        return False
    
    print("\n=== Setup Complete ===")
    print("✓ All setup steps completed successfully!")
    print("\nTo start the application:")
    print("  python manage.py runserver")
    print("\nThen visit: http://127.0.0.1:8000")
    
    return True

if __name__ == "__main__":
    success = setup_project()
    if not success:
        print("\nSetup failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n🎉 Setup completed successfully!")
