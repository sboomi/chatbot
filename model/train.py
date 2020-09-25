import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples

batch_size = 8

dataset = ChatDataset()

# Set model in place
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, 
                        shuffle=True, num_workers=0) #2 if everything is fine else 0

i_size=len(X_train[0])
h_size=8
o_size=len(tags)

model = NeuralNet(i_size, h_size,o_size).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epoch = 1000

for epoch in range(n_epoch):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        #labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels.long())

        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{n_epoch}, loss:{loss.item():.4f}")

print(f"Final loss, loss={loss.item():.4f}")

data = {
    "model_state": model.state_dict(),
    "input_size": i_size,
    "output_size": o_size,
    "hidden_size": h_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"Training complete. File saved to {FILE}")