from torch.utils.data import Dataset
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
