import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

# Define a simple neural network
class DNAEncoderNet(nn.Module):
    def __init__(self):
        super(DNAEncoderNet, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Preprocess the dataset
def preprocess_data(df):
    scaler = MinMaxScaler()
    
    # Assuming the dataset has columns 'Book Name', 'Author Name', 'Rating', 'Price'
    df['Book Name'] = df['Book Name'].apply(text_to_dna)
    df['Author Name'] = df['Author Name'].apply(text_to_dna)
    df['Rating'] = df['Rating'].apply(lambda x: number_to_dna(int(x * 10)))
    df['Price'] = df['Price'].apply(text_to_dna)
    
    inputs = df.apply(lambda row: ''.join([row['Book Name'], row['Author Name'], row['Rating'], row['Price']]), axis=1)
    inputs = inputs.apply(lambda x: [int(char) for char in x.replace('A', '0').replace('T', '1')])
    inputs = scaler.fit_transform(list(inputs))
    
    # Assuming the target is some column, here we use 'Rating' as an example
    targets = df['Rating'].apply(lambda x: [int(char) for char in x.replace('A', '0').replace('T', '1')])
    targets = scaler.fit_transform(list(targets))
    
    return list(inputs), list(targets)

# Training function
def train_model(df, epochs=1000, learning_rate=0.001):
    inputs, targets = preprocess_data(df)
    
    model = DNAEncoderNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for input_data, target_data in zip(inputs, targets):
            input_data = Variable(torch.FloatTensor(input_data))
            target_data = Variable(torch.FloatTensor(target_data))

            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target_data)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(inputs)
        losses.append(avg_loss)

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')

    torch.save(model.state_dict(), 'models/trained_model.pth')

    # Visualize the performance
    animate_performance(losses)

