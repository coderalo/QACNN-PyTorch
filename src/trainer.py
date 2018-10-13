import csv
import glob
import numpy as np
import re
import ruamel_yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from itertools import product
from dataset import create_data_loader
from logger import create_logger
from model import Model
from operator import add
from pathlib import Path
from time import gmtime, strftime
from tqdm import tqdm
from vocab import gen_vocab


class Trainer:
    def __init__(self, config):
        self.p_sent = config.p_sent
        self.p_seq_len = config.p_seq_len
        self.q_seq_len = config.q_seq_len
        self.c_seq_len = config.c_seq_len
        self.epochs = config.epochs
        self.batch_size = config.batch_size

        self.logger = create_logger(name="TRAIN")
        self.data_dir = Path(config.data_dir) / "save"
        self.exp_dir = Path(config.data_dir) / "exp" / config.exp
        self.device = \
            torch.device('cuda:{}'.format(config.device)) \
            if config.device >= 0 else torch.device('cpu')
        config.device = self.device

        self.metrics = ["ACC", "LOSS"]

        self.__cur_epoch = 0
        self.vocab = gen_vocab(
            data_dir=Path(config.data_dir),
            word_freq_path=self.data_dir / "vocab.pkl",
            special_tokens=config.special_tokens,
            size=config.vocab_size)

        self.model = Model(config, self.vocab)

    def train(self):
        self.__checker()
        self.__initialize()
        for e in self.train_bar:
            self.log_stats = {}
            self.model.train()
            self.run_epoch(e, "train")
            self.train_bar.write(self.__display(
                [e+1, "TRAIN"] + [self.stats[key] for key in self.metrics]))
            train_stats = self.stats
            self.model.eval()
            self.run_epoch(e, "valid")
            self.train_bar.write(self.__display(
                [e+1, "VALID"] + [self.stats[key] for key in self.metrics]))
            valid_stats = self.stats
            self.__logging(train_stats, valid_stats)
            self.model.save_state(e+1, self.stats, self.exp_dir / "ckpt")
        self.train_bar.close()

    def run_epoch(self, epoch, mode):
        self.__counter = 0
        self.stats = {key: 0.0 for key in self.metrics}

        ebar = tqdm(
            getattr(self, f"{mode}_data_loader"),
            desc=f"[{mode.upper()}]",
            leave=False,
            position=1)
        sbar = tqdm(
            [0],
            desc=f"[Metric]",
            bar_format="{desc} {postfix}",
            leave=False,
            position=2)

        for b, d in enumerate(ebar):
            self.__counter += d['n_data']
            d['passage'] = d['passage'].to(device=self.device)
            d['question'] = d['question'].to(device=self.device)
            d['choices'] = [
                choice.to(device=self.device)
                for choice in d['choices']]
            d['answer'] = d['answer'].to(device=self.device)

            metrics, predictions = self.run_batch(d, mode)
            self.__metric_dislpay(metrics, sbar)

            for key, val in zip(self.metrics, metrics):
                self.stats[key] += val * d['n_data']

        ebar.close()
        sbar.close()

        for key in self.stats:
            self.stats[key] /= self.__counter

    def run_batch(self, batch, mode):
        if mode == "train":
            o = self.model(
                batch['passage'],
                batch['question'],
                batch['choices'])
            metrics, predictions = self.cal_loss(
                logits=o,
                labels=batch['answer'])
            self.model.zero_grad()
            metrics[1].backward()
            if hasattr(self, "max_grad_norm"):
                self.model.clip_grad(self.max_grad_norm)
            self.model.update()
        elif mode == "valid":
            with torch.no_grad():
                o = self.model(
                    batch['passage'],
                    batch['question'],
                    batch['choices'])
                metrics, predictions = self.cal_loss(
                    logits=o,
                    labels=batch['answer'])
        return metrics, predictions

    def cal_loss(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        _, predictions = torch.max(logits, dim=-1)
        acc = torch.sum(predictions == labels).item() / labels.size(0)
        predictions = predictions.tolist()

        return [acc, loss], predictions

    def __checker(self):
        ckpt_dir = self.exp_dir / "ckpt"
        if hasattr(self, "load_path"):
            self.logger.info(f"[*] Start training from {self.load_path}")
            self.model.load_state(self.load_path)
        elif ckpt_dir.is_dir():
            files = glob.glob(f"{ckpt_dir}/epoch*")
            if files != []:
                files.sort()
                self.__cur_epoch = \
                    int(re.search('\d+', Path(files[-1]).name)[0])
            if self.__cur_epoch > 0:
                if self.__cur_epoch < self.epochs:
                    self.logger.info(
                        f"[*] Resume training (epoch {self.__cur_epoch+1}).")
                    self.model.load_state(files[-1])
                else:
                    while True:
                        retrain = input((
                            "The experiment is complete."
                            "Do you want to re-train the model? y/[N] "))
                        if retrain in ['y', 'Y']:
                            break
                        elif regen in ['n', 'N', '']:
                            self.logger.info("[*] Quit the process...")
                            exit()
                    self.logger.info("[*] Start the experiment.")
        else:
            ckpt_dir.mkdir(parents=True)

    def __display(self, cols):
        return (
            f"[{strftime('%Y-%m-%d %H:%M:%S', gmtime())}] " +
            reduce(
                add, (
                    f"{key:>8} " if type(key) in [str, int]
                    else f"{key:>8.2f} "
                    for key in cols)))

    def __metric_dislpay(self, metrics, bar):
        bar.set_postfix_str(
            '\b\b' + ', '.join([
                f"{metric}: {val:5.2f}"
                for metric, val
                in zip(self.metrics, metrics)]))

    def __logging(self, train_stats, valid_stats):
        log = {}
        for key in self.metrics:
            log[f"TRAIN_{key}"] = f"{train_stats[key]:.2f}"
            log[f"VALID_{key}"] = f"{valid_stats[key]:.2f}"
        self.__log_writer.writerow(log)

    def __initialize(self):
        self.train_bar = tqdm(
            range(self.__cur_epoch, self.epochs),
            total=self.epochs,
            desc='[Total Progress]',
            initial=self.__cur_epoch,
            position=0)
        bar_header = ["EPOCH", "MODE"] + self.metrics
        self.train_bar.write(self.__display(bar_header))

        log_path = self.exp_dir / "log.csv"
        fieldnames = [
            f'{mode}_{metric}' for mode, metric
            in product(["TRAIN", "VALID"], self.metrics)]
        if self.__cur_epoch == 0:
            self.__log_writer = csv.DictWriter(
                log_path.open(mode='w', buffering=1),
                fieldnames=fieldnames)
            self.__log_writer.writeheader()
        else:
            self.__log_writer = csv.DictWriter(
                log_path.open(mode='a', buffering=1),
                fieldnames=fieldnames)

        self.train_data_loader = create_data_loader(
            f"{self.data_dir}/train.pkl",
            self.vocab,
            self.p_sent,
            self.p_seq_len,
            self.q_seq_len,
            self.c_seq_len,
            self.batch_size,
            getattr(self, "debug", "False"))

        self.valid_data_loader = create_data_loader(
            f"{self.data_dir}/valid.pkl",
            self.vocab,
            self.p_sent,
            self.p_seq_len,
            self.q_seq_len,
            self.c_seq_len,
            self.batch_size,
            getattr(self, "debug", "False"))
