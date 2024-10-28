from utils.hugginface import upload_missing_files, download_missing_files
from utils.health import health_check
from dotenv import load_dotenv
import typer

load_dotenv()

app = typer.Typer()

@app.command()
def health():
    """
    Check the health of the environment.
    """

    if not health_check():
        typer.Exit(code=1)


@app.command()
def upload():
    """
    Upload missing files to Hugging Face.
    """

    upload_missing_files(local_dir="checkpoints", hf_subdir="checkpoints")


@app.command()
def download():
    """
    Download missing files from Hugging Face.
    """

    download_missing_files(local_dir="checkpoints", hf_subdir="checkpoints")


if __name__ == "__main__":
    app()
