from utils.hugginface import upload_missing_files, download_missing_files
from utils.health import health_check
from training.train import train
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


@app.command()
def train_model(
    run_id: str = Option(..., help="The run ID for the training session."),
    model_type: str = Option(..., help="The type of model to train. (e.g. 'differential', 'classic')"),
    size: str = Option(..., help="The size of the model to train. (e.g. '10M', '20M', '30M')"),
    batch_size: int = Option(1024, help="The batch size for training."),
    workers: int = Option(4, help="The number of workers for training."),
    lr: float = Option(1e-4, help="The learning rate for training."),
    weight_decay: float = Option(1e-2, help="The weight decay for the optimizer."),
    epochs: int = Option(100, help="The number of epochs for training."),
):
    train(
        run_id=run_id,
        model_type=model_type, # type: ignore
        size=size, # type: ignore
        batch_size=batch_size,
        workers=workers,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
    )


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        print(e)
        typer.Exit(code=1)
