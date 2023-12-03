import datetime
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, AdamW
from transformers import get_linear_schedule_with_warmup

from CustomTextDataset import CustomTextDataset
from TrainerUtils import load_model, run_training
from get_file_paths import get_file_paths
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    # Pad the sequences to the maximum length in the batch
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch


# Ensure you have a GPU (even a small one on a MacBook)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pre-trained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = load_model(device)

book_files = get_file_paths('books-processed')

# Create the dataset
dataset = CustomTextDataset(book_files, tokenizer)
print(f"dataset =>{dataset}")

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=2, shuffle=True,collate_fn=custom_collate_fn)
total_steps_per_epoch = len(data_loader)


print(f"total_steps_per_epoch => {total_steps_per_epoch}")


# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=3e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=-1)
# Calculate the total number of steps per epoch


run_training(device, model, data_loader, optimizer, scheduler)


# Generate a timestamp
#timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Save the model with the timestamp as a suffix
#model.save_pretrained(f"./checkpoints/my_custom_gpt2_model_final_{timestamp}")



