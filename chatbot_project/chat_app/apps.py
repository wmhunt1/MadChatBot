# chat_app/apps.py

from django.apps import AppConfig
import os

class ChatAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chat_app'
    
    # Store the initialized assistant on the CLASS
    chatbot_assistant = None 

    def ready(self):
        # The os.environ check is essential to ensure this runs only once
        if os.environ.get('RUN_MAIN', None) != 'true':
            return
            
        print("Initializing ChatBot Assistant...")
        from .chatbot_core import ChatBotAssistant
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        intents_path = os.path.join(base_dir, 'intents.json')
        model_path = os.path.join(base_dir, 'chatbot_model.pth')
        dimensions_path = os.path.join(base_dir, 'dimensions.json')
        
        assistant = ChatBotAssistant(intents_path)
        
        # ... (Your existing loading/training logic is fine) ...
        if not os.path.exists(model_path):
            print("Model not found. Training new model...")
            assistant.parse_intents()
            assistant.prepare_data()
            assistant.train_model(batch_size=8, lr=0.001, epochs=100)
            assistant.save_model(model_path, dimensions_path)
            print("Model trained and saved.")
        else:
            print("Loading existing model...")
            assistant.parse_intents()
            assistant.load_model(model_path, dimensions_path)
            print("Model loaded successfully.")
            
        # Assign the fully loaded assistant to the CLASS ATTRIBUTE
        ChatAppConfig.chatbot_assistant = assistant