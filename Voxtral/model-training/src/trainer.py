import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model import load_model
from src.dataset import EmotionDataset, collate_fn

def train(config, device):

    # Load the model and processor
    model, processor = load_model(config)

    # Load the dummy model if debbuging
    if config.model.use_dummy_model:
        model = model.to(device)

    # Instantiate train and validation datasets
    train_dataset = EmotionDataset(config.data.train)
    val_dataset   = EmotionDataset(config.data.val)

    # Wrap collate_fn with the processor
    collate = lambda batch: collate_fn(batch, processor)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.training.batch_size,
                              shuffle=True, 
                              collate_fn=collate)
    val_loader   = DataLoader(val_dataset,
                              batch_size=config.training.batch_size,
                              shuffle=False,
                              collate_fn=collate)

    # Initiate AdamW optimizer
    optimizer = AdamW(model.parameters(),
                      lr=config.training.lr,
                      weight_decay=config.training.weight_decay)
    # Total optimiser steps accounts for gradient accumulation
    total_steps = (len(train_loader) // config.training.grad_accum) * config.training.epochs
    # Cosine schedule
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Variable for early stopping
    best_val_loss = float("inf")
    patience = config.training.early_stopping_patience
    patience_counter = 0


    for epoch in range(1, config.training.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # Move only tensor fields to the device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            outputs = model(**batch)
            loss = outputs.loss / config.training.grad_accum
            loss.backward()
            train_loss += outputs.loss.item()

            last_step = (step + 1 == len(train_loader))
            # Perform an optimiser step every grad_accum batches, or on the final batch
            if (step + 1) % config.training.grad_accum == 0 or last_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Compute training loss
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        # Compute validation loss
        avg_val_loss = val_loss / len(val_loader)
        
        # Save & print epoch metrics
        epoch_result = f"Epoch {epoch}/{config.training.epochs} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f}"
        print(epoch_result)

        with open('loss.txt', 'a+') as f:
            f.write(epoch_result + '\n')

        # Checkpoint & Early Stopping
        # model.save_pretrained(f"{config.training.output_dir}/Epoch-{epoch}") # Saves each checkpoint in a folder
        print(f"Saved epoch (val_loss={best_val_loss:.4f})")

        # Save model as new best if its loss is smaller than current best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_pretrained(f"{config.training.output_dir}/best")
            # processor.save_pretrained(f"{config.training.output_dir}/best") # Creates an unsable tokenizer
            print(f"Saved best model (val_loss={best_val_loss:.4f})")
        else:
            # Increment counter and stop early if patience is exhaust
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")