# chat_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # API endpoint (The one we created previously)
    path('chat/', views.chat_api, name='chat_api'),
    
    # New URL for the HTML page (e.g., at the root of the app)
    path('', views.chat_page, name='chat_page'), 
]