import subprocess
import torch
import torch.nn as nn

from datetime import datetime
from logger import create_logger
from net import Net


class Model:
    def __init__(self, config, vocab):
        self._logger = create_logger(name="MODEL")
        self._device = config.device
        self._logger.info("[*] Creating model.")
        self._stats = None
        self._net = Net(config, vocab)
        self._net.to(device=self._device)
        optim = getattr(torch.optim, config.optim)
        self._optim = optim(
            filter(lambda p: p.requires_grad, self._net.parameters()),
            **config.optim_param)

    def train(self):
        self._net.train()

    def eval(self):
        self._net.eval()

    def __call__(self, *args, **kwargs):
        return self._net(*args, **kwargs)

    def infer(self, *args, **kwargs):
        return self._net.inter(*args, **kwargs)

    def zero_grad(self):
        self._optim.zero_grad()

    def clip_grad(self, max_norm):
        nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self._net.parameters()),
            max_norm)

    def update(self):
        self._optim.step()

    def save_state(self, epoch, stats, ckpt_dir):
        ckpt_path = ckpt_dir / f'epoch-{epoch:0>2}.ckpt'
        # self._logger.info("[*] Save model.")
        torch.save({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'stats': stats,
            'net_state': self._net.state_dict(),
            'optim_state': self._optim.state_dict()}, ckpt_path)
        # self._logger.info(f"[-] Model saved to {ckpt_path}.")
        if self.compare(stats, self._stats):
            # self.logger.info("[*] New record achieved!")
            best_ckpt_path = ckpt_dir / 'best.ckpt'
            subprocess.call(["cp", ckpt_path, best_ckpt_path])
            # self.logger.info("[*] Model saved to {best_ckpt_path}")
            self._stats = stats

    def load_state(self, ckpt_path):
        self._logger.info("[*] Load model.")
        ckpt = torch.load(ckpt_path)
        self._net.load_state_dict(ckpt['net_state'])
        self._net.to(self._device)
        self._optim.load_state_dict(ckpt['optim_state'])
        self._stats = ckpt['stats']
        for state in self._optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device=self._device)
        print(f"[-] Model loaded from {ckpt_path}.")

    def load_best_state(self, ckpt_dir):
        ckpt_path = ckpt_dir / 'best.ckpt'
        self.load_state(ckpt_path)

    def compare(self, stats, best_stats):
        if best_stats is None:
            return True
        elif stats['ACC'] > best_stats['ACC']:
            return True
        else:
            return False
