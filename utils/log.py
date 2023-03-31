import os
import logging


def create_logger(output_folder, filename='model_logs.log', file_level=logging.INFO, screen_level=logging.WARNING):
    """

    Args:
        output_folder:
        filename:
        file_level:
        screen_level:

    Returns:
    """
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a logger object
    logger = logging.getLogger(filename)

    # If logging has laready started then return the existing logger
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel(file_level)

    # Add a screen logger
    ch = logging.StreamHandler()
    ch.setLevel(screen_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Next add a file level loggers
    ch = logging.FileHandler(os.path.join(output_folder, filename))
    ch.setLevel(file_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.propagate = False

    return logger
