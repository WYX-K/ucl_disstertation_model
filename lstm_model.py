import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def forward(self, region_sequences, time_sequences):
        # Get embeddings for region sequences
        region_embed = self.region_embedding(region_sequences)
        # Get embeddings for time sequences and multiply by time sequences
        time_embed = self.time_embedding(
            region_sequences) * time_sequences.unsqueeze(-1)
        # Concatenate region and time embeddings
        combined_embed = torch.cat([region_embed, time_embed], dim=-1)
        # Pass through LSTM
        _, (hidden, _) = self.lstm(combined_embed)
        out = self.fc(hidden[-1])
        return out
