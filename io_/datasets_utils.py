import os
from sklearn.model_selection import train_test_split

output_datasets = {0: 'negative', 1: 'positive'}

def split_data(df, path, test_size=0.2):
    train, test = train_test_split(df, test_size=test_size)
    train, dev = train_test_split(train, test_size=test_size)
    train.reset_index(drop=True, inplace=True)
    dev.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    train.index.name = 'index'
    dev.index.name = 'index'
    test.index.name = 'index'
    train.to_csv(os.path.join(path, 'train.csv'))
    dev.to_csv(os.path.join(path, 'dev.csv'))
    test.to_csv(os.path.join(path, 'test.csv'))