from huggingface_hub import HfApi
import os
from typing import Dict
from rich import print
import tempfile
import shutil


def get_local_directory_tree(path: str) -> Dict[str, str]:
    tree = {}
    for dirpath, _, filenames in os.walk(path):
        # Skip .cache directories and hidden directories
        if any(part.startswith('.') for part in dirpath.split(os.sep)):
            continue
            
        relative_path = os.path.relpath(dirpath, path)
        for filename in filenames:
            # Skip hidden files
            if filename.startswith('.'):
                continue
                
            file_path = os.path.join(relative_path, filename).replace("\\", "/")
            if file_path.startswith("./"):
                file_path = file_path[2:]
            tree[file_path] = os.path.join(dirpath, filename)
    return tree


def get_hf_subdirectory_tree(repo_id: str, hf_token: str, subdir: str) -> Dict[str, str]:
    api = HfApi()
    files = api.list_repo_files(repo_id, token=hf_token)

    subdir_files = {}
    for file_path in files:
        # Skip .cache directories and hidden files/directories
        if '.cache' in file_path or any(part.startswith('.') for part in file_path.split('/')):
            continue
            
        if file_path.startswith(subdir):
            relative_path = os.path.relpath(file_path, subdir).replace("\\", "/")
            if relative_path.startswith("./"):
                relative_path = relative_path[2:]
            subdir_files[relative_path] = file_path

    return subdir_files


def upload_missing_files(local_dir: str, hf_subdir: str) -> None:
    hf_repo_id = os.getenv("HF_REPOSITORY")
    hf_token = os.getenv("HF_TOKEN")
    if hf_repo_id is None:
        raise EnvironmentError("HF_REPOSITORY environment variable is not set")
    if hf_token is None:
        raise EnvironmentError("HF_TOKEN environment variable is not set")

    local_tree = get_local_directory_tree(local_dir)
    hf_tree = get_hf_subdirectory_tree(hf_repo_id, hf_token, hf_subdir)

    missing_files = {k: v for k, v in local_tree.items() if k not in hf_tree}

    if not missing_files:
        print("[bold green]All files are already uploaded.[/bold green]")
        return

    api = HfApi()
    for relative_path, full_path in missing_files.items():
        target_path = os.path.join(hf_subdir, relative_path).replace("\\", "/")
        print(f"[bold blue]Uploading {full_path} to {target_path} in HF repository[/bold blue]")
        api.upload_file(
            path_or_fileobj=full_path,
            path_in_repo=target_path,
            repo_id=hf_repo_id,
            token=hf_token,
        )

    print("[bold green]Upload complete.[/bold green]")


def download_missing_files(local_dir: str, hf_subdir: str) -> None:
    """
    Download files from HF repository that are missing in the local directory.

    Args:
        local_dir (str): The local directory to download files to.
        hf_subdir (str): The subdirectory in the HF repository to compare.

    Returns:
        None
    """
    hf_repo_id = os.getenv("HF_REPOSITORY")
    hf_token = os.getenv("HF_TOKEN")
    if hf_repo_id is None:
        raise EnvironmentError("HF_REPOSITORY environment variable is not set")
    if hf_token is None:
        raise EnvironmentError("HF_TOKEN environment variable is not set")

    local_tree = get_local_directory_tree(local_dir)
    hf_tree = get_hf_subdirectory_tree(hf_repo_id, hf_token, hf_subdir)

    missing_files = {k: v for k, v in hf_tree.items() if k not in local_tree}

    if not missing_files:
        print("[bold green]All files are already downloaded.[/bold green]")
        return

    api = HfApi()
    for relative_path, hf_path in missing_files.items():
        # Remove the hf_subdir prefix from the HF path to avoid recursive directories
        target_local_path = os.path.join(local_dir, relative_path)
        target_local_dir = os.path.dirname(target_local_path)
        
        # Skip if file already exists
        if os.path.exists(target_local_path):
            print(f"[bold yellow]Skipping {relative_path} - already exists[/bold yellow]")
            continue
            
        os.makedirs(target_local_dir, exist_ok=True)
        print(f"[bold blue]Downloading {hf_path} to {target_local_dir}[/bold blue]")

        # Create a temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            downloaded_file = api.hf_hub_download(
                repo_id=hf_repo_id,
                filename=hf_path,
                local_dir=temp_dir,
                token=hf_token,
            )
            # Move the file to the correct location
            shutil.move(downloaded_file, target_local_path)

    print("[bold green]Download complete.[/bold green]")
