import datetime
import torch
from transformers import GPT2LMHeadModel


def load_model(device):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(device)
    return model


def run_training(device, model, data_loader, optimizer, scheduler, checkpoint_interval=10):
    num_epochs=3
    total_loss=0.0
    model.train()
    for epoch in range(num_epochs):  # Assuming num_epochs is defined
        for step, batch in enumerate(data_loader, start=1):
            inputs = batch.to(device)
            outputs = model(inputs, labels=inputs)  # Assuming a language model like GPT-2
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            # Print loss every step
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")


            if step % checkpoint_interval == 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                checkpoint_path = f"./checkpoints/my_custom_gpt2_model_{timestamp}_step_{step}_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
    # Print average loss after each epoch
    
    avg_loss = total_loss / len(data_loader)
    print(f"End of Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
            

    # Final model save after training completion
    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    final_model_path = f"./checkpoints/my_custom_gpt2_model_final_{timestamp}.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

