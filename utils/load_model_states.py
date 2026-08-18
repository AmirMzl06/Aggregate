import torch
import os

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    step,
    
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }
    checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    try:

        checkpoint = torch.load(path)
        batch = checkpoint['step']
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return batch
    
    except Exception as e:
        print(e)
        return 0
