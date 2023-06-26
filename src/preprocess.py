from utils.markup_utils import *
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import numpy as np
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')


PATH1 = os.path.join('data','train_part_1')
PATH2 = os.path.join('data','train_part_2')
PATH3 = os.path.join('data','train_part_3')
PATH4 = os.path.join('data','test_ner_only')


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, data, text_vocab_len, target_vocab_len, classes = None,
                 transform=None, target_transform=None):
        self.data = data
        self.target_vocab_len = target_vocab_len
        self.sequence_len = len(data.iloc[0][0])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens, tag = self.data.iloc[idx]
        return torch.Tensor(tokens).int(), torch.Tensor(one_hot(tag, self.target_vocab_len))
    

def make_tl(df, vocab, vocab_lables):
    tokens = []
    lables = []
    max_len = 0
    for ind in df.index:
        tokens.append(vocab(df['word'][ind]))
        lables.append(vocab_lables(df['tag'][ind]))
        if len(df['word'][ind]) > max_len:
            max_len = len(df['word'][ind])
    df['tokens'] = tokens
    df['lables'] = lables
    return df,max_len


def make_pad(df):
    list_sent = []
    list_labels = []
    for ind in df.index:
        list_sent.append(df['tokens'][ind])
        list_labels.append(df['lables'][ind])
    padded_sent = pad_sequences(list_sent)
    padded_labels = pad_sequences(list_labels)
    print('padded_sent shape ', padded_sent.shape)
    padd_df = pd.DataFrame(columns = ['sentence', 'labels'])
    padd_df['sentence'] = pd.Series(padded_sent.tolist())
    padd_df['labels'] = pd.Series(padded_labels.tolist())
    return padd_df


def one_hot(x: np.ndarray, vocab_len: int) -> np.ndarray:
    """
    Args:
        x - одномерный массив значений словаря
        vocab_len - длина словаря
    Выход:
        двумерный массив encoded, где encoded[i] - результат one hot кодирования x[i]
    """
    encoded = np.zeros((len(x), vocab_len))
    for i in range(len(x)):
        encoded[i][x[i]] = 1
    return encoded

def save_vocab(vocab, path):
    with open(path, 'w+', encoding='utf-8') as f:     
        for token, index in vocab.get_stoi().items():
            f.write(f'{index}\t{token}\n')

def read_vocab(path):
    vocab = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            index, token = line.split('\t')
            token = token.replace('\n','')
            vocab[token] = int(index)
    return vocab


if __name__ == "__main__":
    # Filenames without extensions
    filenames = [filename[:-4] for filename in os.listdir(PATH1) if filename[-4:] == '.txt']
    train_df = make_data(PATH1, filenames)

    filenames =  [filename[:-4] for filename in os.listdir(PATH2) if filename[-4:] == '.txt']
    train_df_2 = make_data(PATH2, filenames)
    train_df = pd.concat([train_df, train_df_2], ignore_index=True)
    
    filenames =  [filename[:-4] for filename in os.listdir(PATH3) if filename[-4:] == '.txt']
    train_df_3 = make_data(PATH3, filenames)
    train_df = pd.concat([train_df, train_df_3], ignore_index=True)

    filenames = [filename[:-4] for filename in os.listdir(PATH4) if filename[-4:] == '.txt']
    test_df = make_data(PATH4, filenames)

    train_df, val_df = train_test_split(train_df, test_size=0.1)
    train_df.reset_index(drop = True, inplace= True)
    val_df.reset_index(drop = True, inplace= True)

    vocab = build_vocab_from_iterator(train_df['word'], min_freq=1, specials=["<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])
    vocab_lables = build_vocab_from_iterator(train_df['tag'], min_freq=1)

    train_df,max_len1 = make_tl(train_df,vocab,vocab_lables)
    val_df,max_len2 = make_tl(val_df,vocab,vocab_lables)
    test_df,max_len1 = make_tl(test_df,vocab,vocab_lables)

    train_df = make_pad(train_df)
    val_df = make_pad(val_df)
    test_df = make_pad(test_df)

    text_vocab_len = len(vocab)
    target_vocab_len = len(vocab_lables)

    save_vocab(vocab, 'data/vocab.txt')
    save_vocab(vocab, 'data/vocab_lables.txt')

    train_df.to_csv('data/train_df.csv')
    val_df.to_csv('data/val_df.csv')
    test_df1 = test_df.iloc[:15000,:]
    test_df1.to_csv('data/test_df_1.csv')
    test_df2 = test_df.iloc[15000:,:]
    test_df2.reset_index(inplace= True)
    test_df2.to_csv('data/test_df_2.csv')
