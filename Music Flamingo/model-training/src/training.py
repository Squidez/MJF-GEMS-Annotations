import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .utils import move_to_device

def train_epoch(model, dataloader, optimizer, scheduler, trainable_params, device, epoch):
    model.train()
    total_loss = 0

    for step, batch in enumerate(dataloader):
        batch     = move_to_device(batch, device)
        # batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        #          for k, v in batch.items()}
        outputs   = model(**batch)
        loss      = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if step % 10 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def save_checkpoint(model, processor, config, epoch):
    ckpt_dir = f"{config.output_dir}/epoch-{epoch+1}"
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    print(f"Checkpoint saved : {ckpt_dir}")


def train_model(model, dataloader, processor, config, device):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = AdamW(trainable_params, lr=config.lr, weight_decay=0.01)
    scheduler  = CosineAnnealingLR(optimizer, T_max=config.epochs * len(dataloader))

    for epoch in range(config.epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, trainable_params, device, epoch)
        print(f"Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}")
        save_checkpoint(model, processor, config, epoch)