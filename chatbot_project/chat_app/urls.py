# chat_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # Map the root of the app to the chat_api view
    path('chat/', views.chat_api, name='chat_api'),
]