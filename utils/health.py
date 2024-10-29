from rich.table import Table
from rich import print
import torch
import sys
import os


def config_check() -> bool:
    issues = False

    config_table = Table("Configuration", "Value", show_header=False)

    config_table.add_row("Python", sys.version)
    config_table.add_row("PyTorch", torch.__version__)

    print(config_table)

    return not issues


def gpu_check() -> bool:
    issues = False

    gpu_table = Table("ID", "GPU", show_header=False)

    if torch.cuda.is_available():
        gpu_table = Table("ID", "GPU", show_header=False)

        for i in range(torch.cuda.device_count()):
            gpu_table.add_row(f"GPU {i}", torch.cuda.get_device_name(i))

        print(gpu_table)
    else:
        issues = True
        print("[bold red]No GPUs available![/bold red]")

    return not issues


def env_check() -> bool:
    issues = False

    env_vars = ["HF_REPOSITORY", "HF_TOKEN", "WANDB_API_KEY", "RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
    env_table = Table("Environment Variable", "Value", show_header=False)

    for var in env_vars:
        value = os.getenv(var, None)

        if value is None:
            issues = True
            env_table.add_row(var, "[bold red]NONE[/bold red]")
        else:
            env_table.add_row(var, value)

    print(env_table)

    return not issues


def health_check() -> bool:
    return config_check() and gpu_check() and env_check()
