from transformers import GPT2Tokenizer
import os

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def count_tokens(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)

def count_total_tokens(directory, tokenizer):
    total_tokens = 0
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            total_tokens += count_tokens(file_path, tokenizer)
            print(f"Processed {filename}: Total tokens so far = {total_tokens}")
    return total_tokens

# Path to your 'books' directory
books_dir = '.'  # Update with your directory path
total_tokens = count_total_tokens(books_dir, tokenizer)
print(f"Total number of tokens in all books: {total_tokens}")
