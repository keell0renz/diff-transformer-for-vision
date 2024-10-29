from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.adamw import AdamW
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data.dataloader import default_collate

from utils.model import save_model_to_safetensors
from utils.general import get_train_logger
from training.dataset import TinyImageNet
from training.config import models

from typing import Literal
from rich import print
from tqdm import tqdm
import wandb


def custom_collate(batch):
    try:
        result = default_collate(batch)
        return result
    except Exception:
        raise

def train(
    run_id: str,
    model_type: Literal["classic", "differential"],
    size: Literal["10M", "20M", "30M"],
    batch_size: int = 1024,
    workers: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    epochs: int = 100,
):
    logger = get_train_logger(run_id)

    config = {
        "model_type": model_type,
        "size": size,
        "batch_size": batch_size,
        "workers": workers,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
    }

    wandb.init(project="diff-transformer-for-vision", name=run_id, id=run_id, config=config)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    train_dataset = TinyImageNet(repo="zh-plus/tiny-imagenet", split="train")
    val_dataset = TinyImageNet(repo="zh-plus/tiny-imagenet", split="valid")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=rank
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,  # Small batch size for debugging
        sampler=train_sampler, 
        num_workers=1,  # Single worker
            collate_fn=custom_collate
        )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=custom_collate
    )

    model = models[model_type][size]
    model.cuda()

    model = DDP(model, device_ids=[rank])

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = batch
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"
            ):
                images, labels = batch
                images, labels = images.cuda(), labels.cuda()

                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if rank == 0:
            train_loss = running_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)
            val_acc = 100 * correct / total

            logger.info(
                f"Epoch [{epoch + 1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss_avg:.4f}, Validation Accuracy: {val_acc:.2f}%"
            )
            print(
                f"Epoch [bold green]{epoch + 1}/{epochs}[/bold green], Training Loss: [bold green]{train_loss:.4f}[/bold green], Validation Loss: [bold green]{val_loss_avg:.4f}[/bold green], Validation Accuracy: [bold green]{val_acc:.2f}%[/bold green]"
            )

            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss_avg,
                    "val_acc": val_acc,
                }
            )

    if rank == 0:
        logger.info("Training completed!")
        print("[bold green]Training completed![/bold green]")

    save_model_to_safetensors(model, "./checkpoints/{run_id}/model.safetensors")

    dist.destroy_process_group()
