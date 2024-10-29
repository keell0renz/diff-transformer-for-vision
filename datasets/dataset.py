from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot
from torchvision import transforms
from datasets import load_dataset
from typing import Literal
import torch


class TinyImageNet(Dataset):
    def __init__(self, repo: str, split: Literal["train", "val"]):
        self.dataset = load_dataset(repo, split=split)
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.num_classes = 200

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx):
        sample = self.dataset[idx]  # type: ignore
        image, label = sample["image"], sample["label"]

        image = self.transform(image)

        label_one_hot = one_hot(torch.tensor(label), num_classes=self.num_classes)

        return image, label_one_hot


def get_train_loader(batch_size: int = 1024):
    train_dataset = TinyImageNet(repo="zh-plus/tiny-imagenet", split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_val_loader(batch_size: int = 1024):
    val_dataset = TinyImageNet(repo="zh-plus/tiny-imagenet", split="val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader
