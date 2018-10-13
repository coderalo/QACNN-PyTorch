import argparse
import numpy as np
import torch
import random
import subprocess
import warnings

from box import Box
from logger import create_logger
from pathlib import Path
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-c', '--config', dest='config_path',
            default='./config.yaml', type=Path,
            help='the path of config file')
    args = parser.parse_args()
    return vars(args)


def main(config_path):
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    config = Box.from_yaml(config_path.open())
    logger = create_logger(name="MAIN")
    logger.info(f'[-] Config loaded from {config_path}')
    logger.info(f'[-] Experiment: {config.exp}')

    exp_path = Path(config.data_dir) / "exp" / config.exp
    if not exp_path.is_dir():
        exp_path.mkdir(parents=True)
    subprocess.call(['cp', config_path, exp_path])

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    logger.info('[-] Random seed set to {}'.format(config.random_seed))

    logger.info(f'[*] Initialize trainer...')
    trainer = Trainer(config)
    logger.info('[-] Trainer initialization completed')
    logger.info('[*] Start training...')
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(**args)
