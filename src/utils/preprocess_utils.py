import pandas as pd
import numpy as np
import torch


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


def make_pad(df, pad_sequences):
    list_sent = []
    list_labels = []
    for ind in df.index:
        list_sent.append(df['tokens'][ind])
        list_labels.append(df['lables'][ind])
    padded_sent = pad_sequences(list_sent)
    padded_labels = pad_sequences(list_labels)
    print(padded_sent.shape)
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
