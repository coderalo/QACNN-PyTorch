import codecs
import glob
import torch
import torch.nn as nn

from logger import create_logger


class Embedding(nn.Module):
    def __init__(self, config, vocab):
        vocab_size = len(vocab.word)
        logger = create_logger(name="EMBED")
        super(Embedding, self).__init__()
        unk = vocab.word.unk
        pad = vocab.word.pad
        if hasattr(config, 'embedding_dir'):
            files = glob.glob(config.embedding_dir + "/*")
            flag = False
            for filename in files:
                if str(config.emb_dim) in filename:
                    logger.info(f"[-] Use the data from {filename}.")
                    flag = True
                    break
            if flag:
                cover = 0
                weight = torch.randn(vocab_size, config.emb_dim)
                with codecs.open(filename, 'r', encoding='utf-8') as file:
                    for line in file:
                        data = line.strip().split(' ')
                        word, emb = data[0], list(map(float, data[1:]))
                        token = vocab.word.w2t(word)
                        if token != unk:
                            cover += 1
                            weight[token] = torch.FloatTensor(emb)
                self.model = nn.Embedding.from_pretrained(weight)
                logger.info((
                    f"[-] Coverage: {cover}/{vocab_size} "
                    f"({cover / vocab_size * 100:.2f}%)."))
            else:
                logger.info("[-] Match file not found. Train from scratch.")
        else:
            self.model = nn.Embedding(vocab_size, config.emb_dim)
            logger.info("[-] Train from scratch.")

    def forward(self, i):
        return self.model(i)

