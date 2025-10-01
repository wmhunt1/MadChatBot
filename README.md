# MadChatBot
Sibling Rivalry Chatbot: An NLP Demonstration
This project is a unique demonstration of a natural language processing (NLP) chatbot designed to simulate the antagonistic, yet familiar, communication style of a sibling. Instead of aiming for helpfulness, this chatbot specializes in generating responses characterized by argument, denial, and playful annoyance.

Key Features and Tone
The chatbot is built around specific sibling-centric intents that cover typical interactions:

Argumentative Responses: It automatically refutes user-stated facts or trivia, embodying the classic "I'm right, you're wrong" sibling dynamic.

Denial of Fault: It denies responsibility for messes or broken items, often shifting the blame back onto the user.

Claiming Ownership: It provides quick, selfish reasons for possessing the user's belongings.

Reluctant Compliance: It responds to requests for favors (like doing a chore) with immediate protest or a demand for payment/trade.

Technology Stack
The chatbot is constructed using fundamental machine learning and deep learning tools in Python:

Natural Language Toolkit (NLTK): Used for preprocessing the text data.

Tokenization: Breaking sentences down into individual words.

Lemmatization: Reducing words to their base or root form (e.g., "running" becomes "run") to ensure different tenses of the same word are treated as one feature.

Data Representation (Bag-of-Words - BoW): User input is converted into numerical data using the Bag-of-Words approach. Each pattern is represented as a vector where each index corresponds to a word in the entire vocabulary. A '1' indicates the word's presence, and a '0' indicates its absence. This vector serves as the feature input for the model.

PyTorch Deep Learning Framework: The core intelligence is an Artificial Neural Network built with PyTorch.

Model Architecture: A simple, fully connected (dense) network (ChatBotModel) consisting of multiple linear layers and ReLU activation functions.

Training: The model is trained using the Cross-Entropy Loss function and the Adam optimizer to map the BoW input vectors to the correct intent tag (e.g., deny_fault or greeting_sibling).

Intent Classification: The model's final output is a probability distribution across all defined intents. The intent with the highest probability is selected, and a random, pre-written, and appropriately sarcastic response is chosen from that category.
