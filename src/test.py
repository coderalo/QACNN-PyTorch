import argparse
import csv
import numpy as np
import torch
import random
import subprocess
import warnings

from box import Box
from dataset import create_data_loader
from logger import create_logger
from pathlib import Path
from trainer import Trainer
from model import Model
from vocab import gen_vocab


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-c', '--config', dest='config_path',
            default='./config.yaml', type=Path,
            help='the path of config file')
    parser.add_argument(
            '-o', '--output', dest='output_file',
            default='./output.csv', type=Path,
            help='the path of output file')
    args = parser.parse_args()
    return vars(args)


def main(config_path, output_file):
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    config = Box.from_yaml(config_path.open())
    logger = create_logger(name="MAIN")
    logger.info(f'[-] Config loaded from {config_path}')

    vocab = gen_vocab(
        Path(config.data_dir),
        Path(config.data_dir) / "save" / "vocab.pkl",
        config.special_tokens,
        config.vocab_size)

    model = Model(config, vocab)

    if hasattr(config, "test_ckpt"):
        logger.info(f'[-] Test checkpoint: {config.test_ckpt}')
        model.load_state(config.test_ckpt)
    else:
        logger.info(f'[-] Test experiment: {config.exp}')
        model.load_best_state(Path(config.data_dir) / "exp" / config.exp)

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    logger.info('[-] Random seed set to {}'.format(config.random_seed))

    logger.info(f'[*] Initialize data loader...')
    data_loader = create_data_loader(
        Path(config.data_dir) / "save" / "test.pkl",
        vocab,
        config.p_sent,
        config.p_seq_len,
        config.q_seq_len,
        config.c_seq_len,
        config.batch_size,
        debug=False)
    logger.info('[*] Start testing...')
    writer = csv.DictWriter(
        open(output_file, 'w'),
        fieldnames=['id', 'ans'])
    writer.writeheader()
    for batch in data_loader:
        batch['passage'] = batch['passage'].to(model._device)
        batch['question'] = batch['question'].to(model._device)
        batch['choices'] = batch['choices'].to(model._device)
        logits = model(
            batch['passage'], batch['question'], batch['choices'])
        _, predictions = torch.max(logits, dim=-1)
        predictions = predictions.tolist()
        for ID, pred in zip(batch['id'], predictions):
            writer.writerow({
                'id': ID,
                'ans': int(pred)+1})


if __name__ == "__main__":
    args = parse_args()
    main(**args)
