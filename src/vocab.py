import codecs
import csv
import glob
import json
import pickle

from collections import namedtuple, Counter
from functools import reduce
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tqdm import tqdm
from logger import create_logger


SPT = namedtuple('SPT', ['w', 't', 'o'])


class SpecialVocab:
    def __init__(self, words):
        self._words = words
        for t, w in enumerate(words):
            setattr(self, w, SPT(w=f'<{w}>', t=t, o=f'{w}'))

    def __len__(self):
        return len(self._words)

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx < len(self._words):
            self._idx += 1
            return getattr(self, self._words[self._idx-1])
        raise StopIteration


class WordVocab:
    def __init__(self, word_freq, sp_vocab, size=None, min_cnt=None):
        self._sp_vocab = sp_vocab
        for word in sp_vocab:
            setattr(self, word.o, word.t)
        if size is None:
            # control the size by min count
            if min_cnt is not None:
                for idx, (w, c) in enumerate(word_freq.most_common()):
                    if c < min_cnt:
                        break
                self._size = idx
            # full vocab
            else:
                self._size = len(word_freq)
        else:
            self._size = size
        self._tw = [spt.w for spt in self._sp_vocab] + \
            [w for w, _ in word_freq.most_common(self._size)]
        self._wt = {w: t for t, w in enumerate(self._tw)}
        self._size += len(self._sp_vocab)

    def s2t(self, ws):
        return [self.w2t(w) for w in ws]

    def t2s(self, ts):
        return [self.t2w(t) for t in ts]

    def w2t(self, w):
        return self._wt.get(w, self._sp_vocab.unk.t)

    def t2w(self, t):
        return self._tw[int(t)]

    def __len__(self):
        return self._size

    @property
    def sp(self):
        return [w for w in self._sp_vocab]


def parse_word_freq(data_dir, word_freq_path, logger):
    if word_freq_path.exists():
        """
        while True:
            regen = input(
                "File exists. Do you want to regenerate the data? y/[N] ")
            if regen in ['y', 'Y']:
                word_freq_path.unlink()
                break
            elif regen in ['n', 'N', '']:
                word_freq = pickle.load(word_freq_path.open('rb'))
                break
        """
        word_freq = pickle.load(word_freq_path.open('rb'))
        logger.info(f"[*] Load word frequency data from {word_freq_path}.")
    if not word_freq_path.exists():
        logger.info("[*] Generate word frequency data from scratch.")
        word_freq = Counter()
        files = glob.glob(str(data_dir / "train" / "*.csv.filter"))
        t = tqdm(files, desc='Processing data', leave=False)
        for f in t:
            with codecs.open(f, 'r', encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader, None)
                for line in reader:
                    for s in line[1:-1]:
                        for w in s.strip():
                            word_freq[w] += 1
        if not word_freq_path.parent.is_dir():
            word_freq_path.parent.mkdir()
            logger.info("[*] Directory doesn't exist. Create it now.")
        pickle.dump(word_freq, word_freq_path.open('wb'))
        logger.info(f"[-] Save the word frequency data to {word_freq_path}.")

    return word_freq


def gen_vocab(
        data_dir, word_freq_path, special_tokens, size=None, min_cnt=None):
    logger = create_logger(name="VOCAB")
    special_vocab = SpecialVocab(special_tokens)
    word_freq = parse_word_freq(data_dir, word_freq_path, logger)
    word_vocab = WordVocab(word_freq, special_vocab, size, min_cnt)
    full_vocab = WordVocab(word_freq, special_vocab)
    Vocab = namedtuple('Vocab', ['word', 'full'])
    vocab = Vocab(word=word_vocab, full=full_vocab)
    logger.info(f"[-] Original vocab size: {len(full_vocab)}.")
    logger.info(f"[-] Shrunk vocab size: {len(word_vocab)}.")

    return vocab


if __name__ == "__main__":
    vocab = gen_vocab(
        data_dir=Path("../data/"),
        word_freq_path=Path("../data/save/vocab.pkl"),
        special_tokens=["unk", "pad", "bos", "eos"],
        size=4000)
