import pickle
import torch

from functools import reduce
from operator import add
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from vocab import gen_vocab


class SumDataset(Dataset):
    def __init__(self, data_path):
        self.data = pickle.load(data_path.open('rb'))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def gen_collate_fn(vocab, p_sent, p_seq_len, q_seq_len, c_seq_len):
    unk = vocab.word.unk
    pad = vocab.word.pad

    def padding(seqs, max_seq_len):
        seqs = [seq[:max_seq_len] for seq in seqs]
        seqs = [
            seq + [pad for _ in range(max_seq_len - len(seq))]
            for seq in seqs]
        return seqs

    def collate_fn(batch):
        batch = {
            key: [data[key] for data in batch]
            for key in batch[0].keys()}
        output = {'n_data': len(batch)}
        output['id'] = batch['id']
        if 'answer' in batch:
            output['answer'] = torch.LongTensor(batch['answer'])
        output['passage'] = \
            [padding(p, p_seq_len) for p in batch['passage']]
        output['passage'] = \
            [
                p + [
                        [pad for _ in range(p_seq_len)]
                        for _ in range(p_sent - len(p))]
                for p in output['passage']
            ]
        output['passage'] = [p[:p_sent] for p in output['passage']]
        output['passage'] = torch.LongTensor(output['passage'])
        output['question'] = \
            torch.LongTensor(padding(batch['question'], q_seq_len))
        output['choices'] = torch.LongTensor(
            [padding(c, c_seq_len) for c in batch['choices']])
        output['choices'] = output['choices'].permute(1, 0, 2)
        return output

    return collate_fn


def create_data_loader(
        filename,
        vocab,
        p_sent,
        p_seq_len,
        q_seq_len,
        c_seq_len,
        batch_size,
        debug=False):
    dataset = SumDataset(Path(filename))
    collate_fn = gen_collate_fn(
        vocab, p_sent, p_seq_len, q_seq_len, c_seq_len)
    if debug is True:
        dataset = dataset[:batch_size * 20]
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn)
    return data_loader


if __name__ == "__main__":
    dataset = SumDataset(Path("../data/save/val.pkl"))
    vocab = gen_vocab(
        data_dir=Path("../data/cnn-dailymail"),
        word_freq_path=Path("../data/save/vocab.pkl"),
        special_tokens=["unk", "pad", "bos", "eos"],
        size=30000)
    collate_fn = gen_collate_fn(vocab, 500, 500, True, True)
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn)
    batch = next(iter(data_loader))
    pickle.dump(batch, Path("../data/save/example.pkl").open('wb'))
