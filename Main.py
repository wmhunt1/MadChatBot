import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

# Download necessary NLTK data for tokenization and lemmatization
# Uncomment these if running for the first time
# nltk.download('punkt')
# nltk.download('wordnet')

# --- Neural Network Model Definition ---
class ChatBotModel(nn.Module):
    # Initializes the neural network layers
    def __init__(self, input_size, output_size):
        super(ChatBotModel,self).__init__()
        
        # Define three fully connected (linear) layers
        self.fc1 = nn.Linear(input_size, 128) # First hidden layer
        self.fc2 = nn.Linear(128,64)         # Second hidden layer
        self.fc3 = nn.Linear(64, output_size) # Output layer (size is number of intents)
        
        # Define activation function and dropout layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    # Defines the forward pass of the network
    def forward(self,x):
        # Apply ReLU activation and dropout after the first layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Apply ReLU activation and dropout after the second layer
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (no activation, CrossEntropyLoss handles softmax)
        x = self.fc3(x)
        
        return x

# --- ChatBot Logic and Data Handling Class ---
class ChatBotAssistant:
    
    # Initialize the assistant with paths and optional external functions
    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path
        
        # Data structures to store parsed intent data
        self.documents = []           # List of (tokenized_pattern, intent_tag) pairs
        self.vocabulary = []          # Master list of all unique words (BoW features)
        self.intents = []             # List of all unique intent tags (for indexing)
        self.intents_responses = {}   # Dictionary mapping intent tags to their responses
        
        self.function_mappings = function_mappings # Optional map for specialized actions
        
        # Data for training (Bag-of-Words vectors and intent indices)
        self.X = None # Features (BoW vectors)
        self.y = None # Labels (intent indices)
        
    @staticmethod
    # Tokenizes text and reduces words to their base form (lemmatization)
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        
        return words
        
    # Creates a Bag-of-Words vector for a list of input words
    def bag_of_words(self, words):
        # 1 if the word is in the input, 0 otherwise
        return [1 if word in words else 0 for word in self.vocabulary ]

    # Loads and preprocesses data from the intents JSON file
    def parse_intents(self):
        # Check if the file exists and load data
        if not os.path.exists(self.intents_path):
            print(f"Error: Intents file not found at {self.intents_path}")
            return # Exit the function if file is missing
            
        with open(self.intents_path, 'r') as f:
            intents_data = json.load(f)
        
        # Loop through each intent dictionary
        for intent in intents_data['intents']:
            # Store the intent tag and its responses
            if intent['tag'] not in self.intents:
                self.intents.append(intent['tag'])
                self.intents_responses[intent['tag']] = intent['responses']

            # Process all patterns for the current intent
            for pattern in intent['patterns']:
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, intent['tag'])) # Store words and tag
                
        # Finalize the vocabulary: remove duplicates and sort alphabetically
        self.vocabulary = sorted(list(set(self.vocabulary)))
            
    # Converts documents into numerical training data (X and y)
    def prepare_data(self):
        bags = []
        indices = [] # Intent index used as the label
        
        # Loop through each document (tokenized pattern and its intent tag)
        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words) # Create the BoW vector (features X)
            
            # Find the numerical index of the intent tag (label y)
            intent_index = self.intents.index(document[1])
            
            bags.append(bag)
            indices.append(intent_index)
            
        # Convert lists to NumPy arrays for training
        self.X = np.array(bags)
        self.y = np.array(indices)
        
    # Trains the PyTorch neural network model
    def train_model(self, batch_size, lr, epochs):
        # Convert NumPy data to PyTorch tensors
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long) # Labels must be long type
        
        # Create dataset and DataLoader for batch processing
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Instantiate the model
        self.model = ChatBotModel(self.X.shape[1], len(self.intents))
        
        # Define loss function (Cross Entropy for multi-class classification) and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0
        
            for batch_X, batch_y in loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                running_loss += loss
                
            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")
            
    # Saves the trained model state and dimensions to disk
    def save_model(self, model_path, dimensions_path):
        # Save the model's learned parameters
        torch.save(self.model.state_dict(), model_path)
            
        # Save input/output dimensions needed to load the model later
        with open(dimensions_path, 'w') as f:
            json.dump({ 'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f)
        
    # Loads a saved model and its dimensions
    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
            
        # Instantiate model with saved dimensions and load state dict
        self.model = ChatBotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        
    # Processes a new user message to determine intent and generate a response
    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
            
        # Convert BoW vector to a tensor for the model
        bag_tensor = torch.tensor([bag], dtype=torch.float32)
            
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation for inference
            predictions = self.model(bag_tensor)
                
        # Get the index of the highest prediction score
        predicted_class_index = torch.argmax(predictions, dim=1).item()
        # Map the index back to the intent tag string
        predicted_intent = self.intents[predicted_class_index]
            
        # Check for and execute any mapped external functions
        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                # NOTE: For this simple example, we don't handle the function return value
                self.function_mappings[predicted_intent]()
                    
        # Return a random response from the predicted intent's response list
        if predicted_intent in self.intents_responses and self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None
    
# --- Main Execution Block ---
if __name__ == '__main__':
    # Initialize the assistant (assuming 'intents.json' is in the same directory)
    assistant = ChatBotAssistant('intents.json')
    
    # Data Preparation and Model Training
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8,lr=0.001,epochs=100)
    
    # Save the trained model for later use
    assistant.save_model('chatbot_model.pth', 'dimensions.json')
    
    print("Chatbot is ready. Type /quit to exit.")
    
    # Main chat loop
    while True:
        # Get user input (corrected usage of the input function)
        message = input('What do you have to say to your sibling?')
        
        if message == '/quit':
            break
        
        # Process the message and print the response
        print(assistant.process_message(message))