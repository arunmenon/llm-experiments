import torch
import torch.nn as nn

class GPT2DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(GPT2DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class GPT2Decoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, vocab_size):
        super(GPT2Decoder, self).__init__()
        self.layers = nn.ModuleList([GPT2DecoderLayer(d_model, n_heads) for _ in range(n_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

# Hyperparameters
d_model = 768    # Hidden size
n_heads = 12     # Number of attention heads
n_layers = 12    # Number of decoder layers
vocab_size = 50257  # Vocabulary size

# Instantiate the model
model = GPT2Decoder(d_model, n_heads, n_layers, vocab_size)

# Example input (batch of token IDs)
input_ids = torch.randint(0, vocab_size, (1, 10))  # Batch size of 1, sequence length of 10

# Forward pass
output = model(input_ids)
print(output.shape)  # Should be [1, 10, vocab_size]
