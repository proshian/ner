from sklearn.metrics import f1_score, accuracy_score
import torch
from tqdm import tqdm
from preprocess import read_vocab, TokenDataset
from utils.markup_utils import make_data
from model import CNN_LSTM
import pandas as pd
from utils.postprocess_utils import remove_pad, make_show
from ipymarkup import show_span_box_markup
import os


def make_test(model,dataloader,device):
    all_true_labels = []
    all_preds = []
    inputs_str = []
    acc_list = []
    model.eval()
    for inputs, labels in tqdm(dataloader):
        batch_size, n_words, n_classes = labels.shape
        labels = labels.reshape(-1, n_classes).to(device)
        outputs = model(inputs.to(device))
        _, preds = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        all_true_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
    epoch_acc = accuracy_score(all_preds, all_true_labels)
    epoch_f1 = f1_score(all_preds, all_true_labels, average='macro')
    print('acc: {:.4f}, f1: {:.4f}'.format(epoch_acc, epoch_f1))
    acc_list.append(epoch_acc.tolist())
    return all_true_labels,all_preds

if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  

    # загружаем словарь 
    vocab_lables = read_vocab('data/vocab_lables.txt')
    vocab = read_vocab('data/vocab.txt')

    # загружаем данные
    test_df_1 = pd.read_pickle('data/test_df_1.pkl')
    test_df_2 = pd.read_pickle('data/test_df_2.pkl')
    test_df = pd.concat([test_df_1,test_df_2], ignore_index=True)
    del test_df['index']

    # финальная подготовка данных
    text_vocab_len = 21185
    target_vocab_len = 17
    dataset =  TokenDataset(test_df,text_vocab_len,target_vocab_len)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    # загружаем модель
    model = CNN_LSTM(21185, n_classes = 17).to(device)
    model.load_state_dict(torch.load('weights/cnn_lstm.pth'))
    
    # делаем предсказания
    all_true,all_pred = make_test(model,test_dataloader,device)
    
    # визуализация
    df_test_viz = remove_pad(test_df, vocab,vocab_lables,all_true,all_pred)
    row_idx = 4
    row = df_test_viz['sentence'][row_idx]
    true_lbl = df_test_viz['real_cat'][row_idx]
    pred_lbl = df_test_viz['pred_cat'][row_idx]
    text,spans1 = make_show(row,true_lbl)
    _,spans2 = make_show(row,pred_lbl)
    print('true:')
    show_span_box_markup(text, spans1)
    print('pred:')
    show_span_box_markup(text, spans2)
