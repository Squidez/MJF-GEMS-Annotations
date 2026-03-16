import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model import load_model
from src.utils import move_to_device
from src.dataset import EmotionDataset, collate_fn

def train(config, device):
    model, processor = load_model(config)
    if config.model.use_dummy_model:
        model = model.to(device)

    train_dataset = EmotionDataset(config.data.train)
    val_dataset   = EmotionDataset(config.data.val)

    collate = lambda batch: collate_fn(batch, processor)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_dataset,   batch_size=config.training.batch_size, shuffle=False, collate_fn=collate)

    optimizer = AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    total_steps = (len(train_loader) // config.training.grad_accum) * config.training.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val_loss = float("inf")
    patience = config.training.early_stopping_patience
    patience_counter = 0

    for epoch in range(1, config.training.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            # batch = move_to_device(batch, device)

            outputs = model(**batch)
            loss = outputs.loss / config.training.grad_accum
            loss.backward()
            train_loss += outputs.loss.item()

            last_step = (step + 1 == len(train_loader))
            if (step + 1) % config.training.grad_accum == 0 or last_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch}/{config.training.epochs} │ train_loss: {avg_train_loss:.4f} │ val_loss: {avg_val_loss:.4f}")

        # Checkpoint & Early Stopping
        model.save_pretrained(f"{config.training.output_dir}/Epoch-{epoch}")
        print(f"Saved epoch (val_loss={best_val_loss:.4f})")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_pretrained(f"{config.training.output_dir}/best")
            processor.save_pretrained(f"{config.training.output_dir}/best")
            print(f"Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")