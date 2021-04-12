import numpy as np
import os
import pandas as pd

np.random.seed(0)

MNLI_DATA_DIR = 'data/MNLI/'
if not os.path.isdir(MNLI_DATA_DIR):
    os.mkdir(MNLI_DATA_DIR)

mnli_train = pd.read_json(os.path.join(MNLI_DATA_DIR, 'raw_data', 'multinli_1.0', 'multinli_1.0_train.jsonl'), lines=True)
mnli_test = pd.read_json(os.path.join(MNLI_DATA_DIR, 'raw_data', 'multinli_1.0', 'multinli_1.0_dev_matched.jsonl'), lines=True)
mnli_test_mm = pd.read_json(os.path.join(MNLI_DATA_DIR, 'raw_data', 'multinli_1.0', 'multinli_1.0_dev_mismatched.jsonl'), lines=True)
mnli_test = pd.concat([mnli_test, mnli_test_mm], ignore_index=True)
mnli_test = mnli_test[mnli_test['gold_label'] != '-']
columns = ['sentence1', 'sentence2', 'gold_label', 'genre']
final_columns = ['sentence1', 'sentence2', 'gold_label']
mnli_train = mnli_train[columns]
mnli_train.index.name = 'index'
mnli_test = mnli_test[columns]
mnli_test.index.name = 'index'
train_genres = set(mnli_train['genre'].unique())
all_genres = set(mnli_test['genre'].unique())

for genre in all_genres:
    genre_dir = os.path.join(MNLI_DATA_DIR, genre)
    if not os.path.isdir(genre_dir):
        os.mkdir(genre_dir)
    if genre in train_genres:
        mnli_train_genre_all = mnli_train[mnli_train['genre'] == genre][final_columns]
        train_ids = mnli_train_genre_all.index.values
        np.random.shuffle(train_ids)
        mnli_train_genre = mnli_train_genre_all.loc[train_ids[:-2000]]
        mnli_dev_genre = mnli_train_genre_all.loc[train_ids[-2000:]]
        mnli_train_genre.to_csv(os.path.join(genre_dir, 'train.csv'))
        mnli_dev_genre.to_csv(os.path.join(genre_dir, 'dev.csv'))
    mnli_test_genre = mnli_test[mnli_test['genre'] == genre][final_columns]
    mnli_test_genre.to_csv(os.path.join(genre_dir, 'test.csv'))

snli_train = pd.read_json(os.path.join(MNLI_DATA_DIR, 'raw_data', 'snli_1.0', 'snli_1.0_train.jsonl'), lines=True)
snli_dev = pd.read_json(os.path.join(MNLI_DATA_DIR, 'raw_data', 'snli_1.0', 'snli_1.0_dev.jsonl'), lines=True)
snli_test = pd.read_json(os.path.join(MNLI_DATA_DIR, 'raw_data', 'snli_1.0', 'snli_1.0_test.jsonl'), lines=True)

snli_train = snli_train[snli_train['gold_label'] != '-']
snli_dev = snli_dev[snli_dev['gold_label'] != '-']
snli_test = snli_test[snli_test['gold_label'] != '-']

columns = ['sentence1', 'sentence2', 'gold_label', 'genre']
final_columns = ['sentence1', 'sentence2', 'gold_label']
snli_train = snli_train[final_columns]
snli_train.index.name = 'index'
snli_dev = snli_dev[final_columns]
snli_dev.index.name = 'index'
snli_test = snli_test[final_columns]
snli_test.index.name = 'index'

genre_dir = os.path.join(MNLI_DATA_DIR, 'captions')
if not os.path.isdir(genre_dir):
    os.mkdir(genre_dir)
snli_train.to_csv(os.path.join(genre_dir, 'train.csv'))
snli_dev.to_csv(os.path.join(genre_dir, 'dev.csv'))
snli_test.to_csv(os.path.join(genre_dir, 'test.csv'))