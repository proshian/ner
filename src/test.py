from sklearn.metrics import f1_score, accuracy_score
import torch
import tqdm
from preprocess import read_vocab, TokenDataset, make_tl, make_pad
from utils.markup_utils import make_data
from model import CNN_LSTM
import pandas as pd
from utils.postprocess_utils import remove_pad, make_show
from ipymarkup import show_span_box_markup
import os
PATH4 = os.path.join('data','test_ner_only')


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
    vocab_lables = read_vocab('data/vocab_labels.txt')
    vocab = read_vocab('data/vocab.txt')

    # загружаем данные
    filenames = [filename[:-4] for filename in os.listdir(PATH4) if filename[-4:] == '.txt']
    test_df = make_data(PATH4, filenames)
    test_df,max_len1 = make_tl(test_df,vocab,vocab_lables)
    test_df = make_pad(test_df)

    # финальная подготовка данных
    dataset =  TokenDataset(test_df)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    # загружаем модель
    model = CNN_LSTM(len(vocab), n_classes = len(vocab_lables)).to(device)
    model.load_state_dict(torch.load('weight/cnn_lstm.pth'))

    all_true,all_pred = make_test(model,test_dataloader,dataset)
    
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
