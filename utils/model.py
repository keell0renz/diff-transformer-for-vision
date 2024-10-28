from safetensors.torch import save_file, load_file
from typing import Dict
import torch
import os


def save_model_to_safetensors(model: torch.nn.Module, path: str) -> None:
    """
    Save a PyTorch model's state dict to a file using safetensors format.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): The path to save the model to.

    Returns:
        None
    """

    state_dict = model.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(state_dict, path)


def load_model_from_safetensors(
    path: str, device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load a PyTorch model's state dict from a safetensors file.

    Args:
        path (str): The path to the safetensors file.
        device (str): The device to load the tensors onto. Defaults to CPU.

    Returns:
        Dict[str, torch.Tensor]: The state dict of the PyTorch model.
    """

    if not os.path.isfile(path):
        raise ValueError(f"{path} is not a valid file.")

    state_dict = load_file(path, device=device)
    return state_dict
