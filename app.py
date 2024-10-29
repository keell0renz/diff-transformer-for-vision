from utils.hugginface import upload_missing_files, download_missing_files
from utils.health import health_check
from models.config import models
from torchsummary import summary
from dotenv import load_dotenv
from typer import Option
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


@app.command()
def model_summary(
    model: str = Option(..., help="The model to summarize."),
    size: str = Option(..., help="The size of the model to summarize."),
):
    try:
        summary(models[model][size].cuda(), (3, 64, 64), device="cuda")
    except (RuntimeError, AssertionError) as e:
        print(e)


if __name__ == "__main__":
    app()
