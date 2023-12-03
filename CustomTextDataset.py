import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa  # Ensure pyarrow is imported

import pickle

class CustomTextDataset(Dataset):
    def __init__(self, file_paths, tokenizer, save_dir='tokenized_data', file_name='tokenized_data.parquet'):
        self.tokenized_texts = []
        self.tokenizer = tokenizer
        self.save_dir = save_dir
        self.file_name = file_name
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(self.save_dir, self.file_name)
        if os.path.exists(save_path):
            self.load_tokenized_data(save_path)
            print(f"Loaded tokenized data from {save_path}")
        else:
            for path in file_paths:
                print(f"Processing file: {path}")
                with open(path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.process_and_add(text)
            self.save_tokenized_data(save_path)
            print(f"Tokenized data processed and saved to {save_path}")


    def process_and_add(self, text):
        sentences = text.split('.')  # Splitting by sentences
        print(f"Number of sentences found: {len(sentences)}")
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Test adding this sentence to the current chunk
            test_chunk = f"{current_chunk} {sentence}." if current_chunk else f"{sentence}."
            test_chunk_tokenized = self.tokenizer.encode(test_chunk, add_special_tokens=False)
            
            if len(test_chunk_tokenized) <= self.tokenizer.model_max_length:
                current_chunk = test_chunk
            else:
                # Add the current chunk and start a new one with the current sentence
                if current_chunk:
                    self.tokenized_texts.append(self.tokenizer.encode(current_chunk, max_length=self.tokenizer.model_max_length, truncation=True))
                current_chunk = f"{sentence}."

        # Add the last chunk
        if current_chunk:
            self.tokenized_texts.append(self.tokenizer.encode(current_chunk, max_length=self.tokenizer.model_max_length, truncation=True))

   
    def save_tokenized_data(self, save_path):
        # Serialize each list of tokens and store in a DataFrame
        serialized_token_chunks = [pickle.dumps(token_chunk) for token_chunk in self.tokenized_texts]
        df = pd.DataFrame({'tokenized_text': serialized_token_chunks})
        table = pa.Table.from_pandas(df)  # Correctly reference pyarrow's Table
        pq.write_table(table, save_path)
        print(f"Tokenized data saved to {save_path}")
        

    def load_tokenized_data(self, load_path):
        table = pq.read_table(load_path)
        df = table.to_pandas()
        self.tokenized_texts = [pickle.loads(blob) for blob in df['tokenized_text']]
        print(f"Loaded {len(self.tokenized_texts)} items from the saved data.")

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return torch.tensor(self.tokenized_texts[idx], dtype=torch.long)

# Example usage
# from transformers import GPT2Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# dataset = CustomTextDataset(file_paths, tokenizer)