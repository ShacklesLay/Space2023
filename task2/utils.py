import logging
import os

import numpy.random
import torch.cuda
import random


def set_logger(log_dir, model_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        file_handler = logging.FileHandler(log_dir)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)