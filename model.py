import torch
import torch.nn as nn

class TextToPointCloud(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_points, point_dim):
        super(TextToPointCloud, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, num_points * point_dim)
        self.num_points = num_points
        self.point_dim = point_dim

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        x = self.fc1(hidden.squeeze(0))
        x = torch.relu(x)
        x = self.fc2(x)
        return x.view(-1, self.num_points, self.point_dim)
