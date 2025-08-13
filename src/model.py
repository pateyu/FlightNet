import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        # use a single head for the final output.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        # Apply dropout to the input features
        x = F.dropout(x, p=0.6, training=self.training)
        # First GAT layer with ELU activation
        x = F.elu(self.conv1(x, edge_index))
        # Dropout on the embeddings
        x = F.dropout(x, p=0.6, training=self.training)
        # Second GAT layer (output layer)
        x = self.conv2(x, edge_index)
        return x