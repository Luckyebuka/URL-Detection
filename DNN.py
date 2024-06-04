import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from DATAset import combined_features


class DeepNeuralNetwork(nn.Module):
    def __init__(self, n_features, n_hidden1, n_hidden2, n_classes):
        super(DeepNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden1)  # Input layer to hidden layer 1
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)  # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(n_hidden2, n_classes)  # Hidden layer 2 to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for hidden layer 1
        x = F.relu(self.fc2(x))  # Activation function for hidden layer 2
        x = self.fc3(x)  # No activation function for output layer
        return x


X = combined_features.drop(['url', 'type', 'tokenized_url', 'lemmatized_url', 'lemmatized_url_str', 'Category'],
                           axis=1).values

y = combined_features['Category'].values

# standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

n_features = X_train.shape[1]
n_hidden1 = 64
n_hidden2 = 64
n_classes = len(torch.unique(y_train_tensor))

model = DeepNeuralNetwork(n_features, n_hidden1, n_hidden2, n_classes)

criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on the testing dataset: {accuracy:.2f}%')
