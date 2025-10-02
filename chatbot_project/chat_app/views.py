# chat_app/views.py

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
# ...
from .apps import ChatAppConfig # Keep this import

@csrf_exempt
def chat_api(request):
    # Retrieve the initialized assistant from the CLASS ATTRIBUTE
    assistant = ChatAppConfig.chatbot_assistant
    
    # Ensure the model is loaded before processing
    if not assistant or not assistant.model: # The error is coming from here!
        return JsonResponse({"error": "Chatbot model not initialized."}, status=503)

    if request.method == 'POST':
        try:
            # ... (rest of your logic)
            data = json.loads(request.body.decode('utf-8'))
            user_message = data.get('message', '').strip()
            
            # Process the message using the chatbot logic
            bot_response = assistant.process_message(user_message)
            
            # ... (rest of the response logic)
            if bot_response is None:
                bot_response = "I'm not sure how to respond to that."

            return JsonResponse({"message": user_message, "response": bot_response})
            
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format."}, status=400)
        except Exception as e:
            print(f"Error processing message: {e}")
            return JsonResponse({"error": "An internal error occurred."}, status=500)
    else:
        return JsonResponse({"info": "Send a POST request with a 'message' in the body to chat."}, status=200)
    
    
from django.shortcuts import render 

def chat_page(request):
    """Renders the main chatbot HTML page."""
    # This automatically looks for chat_app/chat.html in the templates folder
    return render(request, 'chat_app/chat.html')