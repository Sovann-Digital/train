import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
import os
import json
import nltk
nltk.download('punkt')


# Define the directory containing JSON files
data_directory = "./Resources/data/json/"

# Get a list of JSON files in the directory
json_files = [file for file in os.listdir(data_directory) if file.endswith('.json')]

# Initialize lists to store data
all_words = []
tags = []
xy = []

# Loop through each JSON file
for json_file in json_files:
    # Open the JSON file
    with open(os.path.join(data_directory, json_file), 'r', encoding='utf-8') as file:
        # Load JSON data
        intents = json.load(file)
        
        # Process each intent in the JSON data
        for intent in intents:
            tag = intent['instruction']
            tags.append(tag)
            patterns = intent['output'].split('\n\n')
            for pattern in patterns:
                words = tokenize(pattern)
                all_words.extend(words)
                xy.append((words, tag))

# Preprocess words
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []
for (pattern_words, tag) in xy:
    bag = bag_of_words(pattern_words, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
num_epochs = 5000
batch_size = 16
learning_rate = 0.0001
input_size = len(X_train[0])
hidden_size = 16
output_size = len(tags)

# Define Dataset class
class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.n_samples = len(X_data)
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Create DataLoader
dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

# Save the model
MODEL_PATH = "Resources/model/data.pth"
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}
torch.save(data, MODEL_PATH)

print(f'Training complete. Model saved to {MODEL_PATH}')
