import numpy as np

# Generating 20 realistic sample sentences
realistic_sentences = [
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is transforming the world",
    "Python is a popular programming language",
    "The weather today is sunny with a chance of rain",
    "Data science involves statistics and machine learning",
    "Music brings joy to many people",
    "Reading books can be very enlightening",
    "A healthy diet contributes to overall well-being",
    "Regular exercise is key to maintaining good health",
    "Technology is evolving at a rapid pace",
    "Learning new languages opens up many opportunities",
    "The global economy is facing unprecedented challenges",
    "Sustainable practices help protect the environment",
    "Meditation is beneficial for mental health",
    "The universe is vast and full of mysteries",
    "History teaches us about the past",
    "Cooking at home can be a fun activity",
    "Traveling allows you to explore new cultures",
    "Education is fundamental to success",
    "Creativity is essential in art and innovation"
]

# Tokenizing the sentences
tokenized_realistic_sentences = [sentence.lower().split() for sentence in realistic_sentences]

# Simulating word embeddings for the vocabulary
realistic_vocabulary = set(word for sentence in tokenized_realistic_sentences for word in sentence)
word_to_embedding_realistic_32d = {word: np.random.rand(32) for word in realistic_vocabulary}

# Finding the maximum sentence length for padding
max_length_realistic = max(len(sentence) for sentence in tokenized_realistic_sentences)

# Creating the batch with padding for shorter sentences using 32-dimensional embeddings
realistic_batch_32d = np.array([np.vstack([word_to_embedding_realistic_32d[word] for word in sentence] + 
                                           [np.zeros(32)] * (max_length_realistic - len(sentence))) for sentence in tokenized_realistic_sentences])

print("Batch shape:", realistic_batch_32d.shape)


# Simplified causal attention head
def simplified_causal_attention_head(input_embeddings, weight_matrix, mask):
    attention_scores = np.dot(input_embeddings, weight_matrix)
    return np.where(mask, attention_scores, np.zeros_like(attention_scores))

# Multi-head causal attention
def multi_head_causal_attention(input_embeddings, num_heads=2):
    head_outputs = np.zeros_like(input_embeddings)
    sequence_length = input_embeddings.shape[1]
    
    # Mask for causal attention
    mask = np.tril(np.ones((sequence_length, sequence_length), dtype=bool))

    for _ in range(num_heads):
        weight_matrix = np.random.rand(input_embeddings.shape[-1], input_embeddings.shape[-1])
        head_output = simplified_causal_attention_head(input_embeddings, weight_matrix, mask)
        head_outputs += head_output

    return head_outputs / num_heads

# Applying causal attention to the batch
transformed_batch_causal = multi_head_causal_attention(realistic_batch_32d)

# Output the shape of the transformed batch
print("Transformed batch shape:", transformed_batch_causal.shape)


# Number of layers
num_layers = 4  # For example

# Applying multiple layers to the batch
layer_output = realistic_batch_32d
for _ in range(num_layers):
    layer_output = multi_head_causal_attention(layer_output)

# Output the shape of the final layer's output
print("Output shape after multiple layers:", layer_output.shape)
