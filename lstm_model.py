import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
# Define the LSTM model


class LSTM(nn.Module):
    def __init__(self, vocab_size, region_embedding_dim, time_embedding_dim, hidden_size, output_size, num_layers=1, bidirectional=False):
        super(LSTM, self).__init__()
        # Embedding layer for region sequences
        self.region_embedding = nn.Embedding(
            vocab_size, region_embedding_dim, padding_idx=0)
        # Embedding layer for time sequences
        self.time_embedding = nn.Embedding(
            vocab_size, time_embedding_dim, padding_idx=0)
        # LSTM layer
        self.lstm = nn.LSTM(region_embedding_dim +
                            time_embedding_dim, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=bidirectional)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    # Forward pass
    def forward(self, region_sequences, time_sequences, original_length):
        # Get embeddings for region sequences
        region_embed = self.region_embedding(region_sequences)
        # Get embeddings for time sequences and multiply by time sequences
        time_embed = self.time_embedding(
            region_sequences) * time_sequences.unsqueeze(-1)
        # Concatenate region and time embeddings
        combined_embed = torch.cat([region_embed, time_embed], dim=-1)
        combined_embed_packed = pack_padded_sequence(combined_embed, original_length, batch_first=True, enforce_sorted=False)
        # Pass through LSTM
        _, (hidden, _) = self.lstm(combined_embed_packed)
        out = self.fc(hidden[-1])
        return out

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.region_sequences = X[:,2::2]
        self.time_sequences = X[:,3::2]
        self.other_features = X[:,:2]
        self.y = y

    def __getitem__(self, index):
        region_sequences = self.region_sequences[index]
        time_sequences = self.time_sequences[index]
        other_features = self.other_features[index]
        y = self.y[index]
        return region_sequences, time_sequences, other_features, y

    def __len__(self):
        return len(self.region_sequences)