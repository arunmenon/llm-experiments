import torch
import torch.nn as nn

from TransformerBlock import TransformerBlock

class GPT(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout):
        super(GPT, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x

# Parameters for the model
embed_size = 512  # Size of each word embedding
num_layers = 6    # Number of Transformer blocks in the model
heads = 8         # Number of heads in the multi-head attention mechanism
forward_expansion = 4
dropout = 0.1

# Instantiate the model
model = GPT(embed_size, num_layers, heads, forward_expansion, dropout)

# Example input (batch of tokenized sentences)
# Assume `input_data` is a tensor of shape [batch_size, seq_length, embed_size]
# Assume `mask` is a tensor indicating which positions are valid (not padding)
# Example input data (batch_size, seq_length, embed_size)
input_data = torch.rand(64, 10, embed_size)  # Random data for demonstration
mask = None  # Replace with a real mask when using

# Forward pass through the model
output = model(input_data, mask)

print("Output shape:", output.shape)
output = model(input_data, mask)
