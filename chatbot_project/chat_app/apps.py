# chat_app/apps.py

from django.apps import AppConfig
import os

class ChatAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chat_app'
    
    # Store the initialized assistant here
    chatbot_assistant = None 

    def ready(self):
        # Only run this once, during the initial server startup
        if os.environ.get('RUN_MAIN', None) != 'true':
            return
            
        print("Initializing ChatBot Assistant...")
        from .chatbot_core import ChatBotAssistant
        
        # Define paths relative to the app directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        intents_path = os.path.join(base_dir, 'intents.json')
        model_path = os.path.join(base_dir, 'chatbot_model.pth')
        dimensions_path = os.path.join(base_dir, 'dimensions.json')
        
        self.chatbot_assistant = ChatBotAssistant(intents_path)
        
        # Check if the trained model exists to decide whether to train or load
        if not os.path.exists(model_path):
            print("Model not found. Training new model...")
            self.chatbot_assistant.parse_intents()
            self.chatbot_assistant.prepare_data()
            # NOTE: Consider moving training to an external script or management command
            # Training a model in AppConfig is generally discouraged for large models
            # but is acceptable for a simple deployment test.
            self.chatbot_assistant.train_model(batch_size=8, lr=0.001, epochs=100)
            self.chatbot_assistant.save_model(model_path, dimensions_path)
            print("Model trained and saved.")
        else:
            print("Loading existing model...")
            self.chatbot_assistant.parse_intents() # Still need to load intents, vocab, etc.
            # You must call parse_intents before load_model to get the full vocab and intent list
            self.chatbot_assistant.load_model(model_path, dimensions_path)
            print("Model loaded successfully.")