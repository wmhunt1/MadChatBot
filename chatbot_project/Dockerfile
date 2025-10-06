# Use the official Python image as the base image for stability and size
# We choose a slim version for smaller image size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE=chatbot_project.settings

# Set the working directory inside the container
WORKDIR /app

# 1. Install Python Dependencies
# Copy requirements file first to take advantage of Docker layer caching
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 2. Download NLTK Resources
# The chatbot relies on 'punkt' and 'wordnet' for tokenization/lemmatization.
# We download these explicitly here so the app doesn't try to download them on startup.
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# 3. Copy Application Code
# Copy the rest of the application code into the container
COPY . /app/

# 4. Handle Model Training/Loading
# Since the ChatBotAssistant training/loading is integrated into chat_app/apps.py's ready()
# method, it will automatically run when 'gunicorn' starts the main Django process.
# We don't need a separate training command here, but we ensure the files are available.
# (The model files will be created/loaded inside the container's /app/chat_app directory)

# 5. Run the server using Gunicorn
# Gunicorn is a production-ready WSGI server. 
# We run it using the application entry point (the main project WSGI file).
# Using 0.0.0.0 binds to all network interfaces.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "chatbot_project.wsgi:application"]