import logging
import os


def get_train_logger(directory: str) -> logging.Logger:
    """
    Args:
        directory (str): The directory to save the logs to.

    Returns:
        logging.Logger: A logger for training logs.
    """

    log_file = f"./checkpoints/{directory}/training.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
