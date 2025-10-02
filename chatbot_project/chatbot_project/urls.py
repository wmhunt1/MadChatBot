# chatbot_project/urls.py

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # Include the URLs from the chat_app at the 'api/' path
    path('api/', include('chat_app.urls')),
]