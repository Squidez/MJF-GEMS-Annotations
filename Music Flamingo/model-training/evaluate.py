import torch
import json
from src.utils import move_to_device

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            batch   = move_to_device(batch, device)
            outputs = model(**batch)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss