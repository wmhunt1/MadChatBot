# chat_app/views.py

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .apps import ChatAppConfig

@csrf_exempt # Required for simple POST requests without a CSRF token
def chat_api(request):
    # Retrieve the initialized assistant from the AppConfig
    assistant = ChatAppConfig.chatbot_assistant
    
    # Ensure the model is loaded before processing
    if not assistant or not assistant.model:
        return JsonResponse({"error": "Chatbot model not initialized."}, status=503)

    if request.method == 'POST':
        try:
            # Parse the JSON payload from the request body
            data = json.loads(request.body.decode('utf-8'))
            user_message = data.get('message', '').strip()
            
            if not user_message:
                 return JsonResponse({"response": "Please provide a message."}, status=400)

            # Process the message using the chatbot logic
            bot_response = assistant.process_message(user_message)
            
            if bot_response is None:
                bot_response = "I'm not sure how to respond to that."

            return JsonResponse({"message": user_message, "response": bot_response})
            
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format."}, status=400)
        except Exception as e:
            # Log the error for debugging
            print(f"Error processing message: {e}")
            return JsonResponse({"error": "An internal error occurred."}, status=500)
    else:
        # Handle other HTTP methods (e.g., GET)
        return JsonResponse({"info": "Send a POST request with a 'message' in the body to chat."}, status=200)