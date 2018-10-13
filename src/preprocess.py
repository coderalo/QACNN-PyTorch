import argparse
import codecs
import csv
import glob
import json
import pickle
import re

from box import Box
from functools import reduce
from num2chinese import num2chinese
from pathlib import Path
from tqdm import tqdm
from logger import create_logger
from operator import add
from vocab import gen_vocab


def num_replace(match):
    match = match.group()
    return num2chinese(match)


def preprocess(config):
    logger = create_logger(name="DATA")
    data_dir = Path(config.data_dir)
    save_dir = Path(config.data_dir) / "save"
    if not save_dir.is_dir():
        save_dir.mkdir()
        logger.info("[*] Directory doesn't exist. Create it now.")

    train_files = glob.glob(str(data_dir / "train" / "*.csv"))
    val_files = glob.glob(str(data_dir / "valid" / "*.csv"))
    test_files = glob.glob(str(data_dir / "test" / "*.csv"))
    logger.info("[*] Start pre-parsing file.")

    for filename in train_files + val_files + test_files:
        file_preparse(filename)

    vocab_path = save_dir / "vocab.pkl"

    vocab = gen_vocab(
        data_dir,
        vocab_path,
        config.special_tokens,
        size=getattr(config, 'vocab_size', None),
        min_cnt=getattr(config, 'min_cnt', None))

    logger.info("[*] Start tokenizing data...")
    train_files = glob.glob(str(data_dir / "train" / "*.csv.filter"))
    val_files = glob.glob(str(data_dir / "valid" / "*.csv.filter"))
    test_files = glob.glob(str(data_dir / "test" / "*.csv.filter"))

    train_path = save_dir / f"train.pkl"
    train_data = [
        file_preprocess(filename, vocab)
        for filename in tqdm(train_files, desc='TRAIN', leave=False)]
    train_data = reduce(add, train_data)
    pickle.dump(train_data, train_path.open('wb'))
    logger.info(f"[-] Write processed train data to {train_path}.")

    val_path = save_dir / "valid.pkl"
    val_data = [
        file_preprocess(filename, vocab)
        for filename in tqdm(val_files, desc='VALID', leave=False)]
    val_data = reduce(add, val_data)
    pickle.dump(val_data, val_path.open('wb'))
    logger.info(f"[-] Write processed val data to {val_path}.")

    test_path = save_dir / "test.pkl"
    test_data = [
        file_preprocess(filename, vocab)
        for filename in tqdm(test_files, desc='TEST ', leave=False)]
    test_data = reduce(add, test_data)
    pickle.dump(test_data, test_path.open('wb'))
    logger.info(f"[-] Write processed test data to {test_path}.")


def file_preparse(filename):
    with codecs.open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [line for line in reader]

    for i, d in enumerate(data):
        for j, s in enumerate(d[1:7]):
            s = re.sub('\d+', num_replace, s)
            s = re.sub("[\t\n 　]", '', s)
            s = re.sub("[~～]?((tape)|(TAPE))", '', s)
            s = re.sub("~VOICE", '', s)
            pattern = \
                u"[\u4e00-\u9fff，。！？：（）!:,「」、“”.《》?%();；﹐﹑％]+"
            s = 'U'.join(re.findall(pattern, s))
            data[i][j+1] = s

    with codecs.open(filename + ".filter", 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for d in data:
            writer.writerow(d)


def file_preprocess(filename, vocab):
    with codecs.open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)
        text = [line for line in reader]
    data = []
    pattern = '，|。|！|？|,|!|;|:|：|、|；|「|」|\.|﹐'
    for QA in text:
        d = {}
        d['id'] = str(QA[0])
        d['passage'] = [
            vocab.word.s2t(s)
            for s in re.split(pattern, QA[1])
            if s != '']
        d['question'] = vocab.word.s2t(QA[2])
        d['choices'] = \
            [
                vocab.word.s2t(QA[3]),
                vocab.word.s2t(QA[4]),
                vocab.word.s2t(QA[5]),
                vocab.word.s2t(QA[6])
            ]
        if len(QA) >= 8:
            d['answer'] = int(re.findall('\d+', QA[7])[0]) - 1
        data.append(d)
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', dest='config_path',
        default='./config.yaml', type=Path,
        help='the path of the config file')
    args = parser.parse_args()
    return vars(args)


def main(config_path):
    config = Box.from_yaml(config_path.open())
    preprocess(config)


if __name__ == "__main__":
    args = parse_args()
    main(**args)
