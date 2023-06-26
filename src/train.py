import tqdm
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import Adam, AdamW
from model import CNN_LSTM
from torch import nn
import matplotlib.pyplot as plt
from preprocess import read_vocab, TokenDataset
import pandas as pd

def train_model(model, criterion, optimizer, num_epochs, dataloader, device):
    all_true_labels = []
    all_preds = []
    inputs_str = []
    loss_list = {'train' : [], 'val':[]}
    acc_list = {'train' : [], 'val':[]}
    f1_list = {'train' : [], 'val':[]}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                print('start train')
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_true_labels = []
            all_preds = []
            inputs_str = []
            for inputs, labels in tqdm(dataloader[phase]):
                batch_size, n_words, n_classes = labels.shape
                labels = labels.reshape(-1, n_classes).to(device)
                outputs = model(inputs.to(device))
                
                loss = criterion(outputs, labels)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() #* inputs.size(0)
                _, labels = torch.max(labels, 1)

                all_true_labels.extend(labels.tolist())
                all_preds.extend(preds.tolist())
            epoch_loss = running_loss /len(dataloader[phase])
            epoch_acc = accuracy_score(all_preds, all_true_labels)
            epoch_f1 = f1_score(all_preds, all_true_labels, average='macro')
            print('{} loss: {:.4f}, acc: {:.4f}, f1: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc,
                                                        epoch_f1            
                                                        ))
            loss_list[phase].append(epoch_loss)
            acc_list[phase].append(epoch_acc.tolist())
            f1_list[phase].append(epoch_f1.tolist())
    return all_true_labels, all_preds, inputs_str, loss_list, acc_list, f1_list


def graf(loss, acc, f1, epoch_num):
    epox_list = [i for i in range(epoch_num)]
    fig, ax = plt.subplots(2, 3, figsize=(26, 13))
    ax[0, 0].plot(epox_list, loss['train'])
    ax[0, 0].set_title("Изменение потерь на обучающей выборке")
    ax[0, 1].plot(epox_list, acc['train'])
    ax[0, 1].set_title("Изменение точности на обучающей выборке")
    ax[0, 2].plot(epox_list, f1['train'])
    ax[0, 2].set_title("Изменение f1-score на обучающей выборке")
    ax[1, 0].plot(epox_list, loss['val'])
    ax[1, 0].set_title("Изменение потерь на валидационной выборке")
    ax[1, 1].plot(epox_list, acc['val'])
    ax[1, 1].set_title("Изменение точности на валидационной выборке")
    ax[1, 2].plot(epox_list, f1['val'])
    ax[1, 2].set_title("Изменение f1-score на валидационной выборке")
    plt.show()



if __name__ == "__main__":
    epoch_num = 20
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    vocab_lables = read_vocab('data/vocab_labels.txt')
    vocab = read_vocab('data/vocab.txt')
    n_classes = len(vocab_lables)

    train_df = pd.read_pickle('data/train_df.pkl')
    val_df = pd.read_pickle('data/val_df.pkl')

    text_vocab_len = len(vocab)
    target_vocab_len = len(vocab_lables)
    datasets = {
    'train': TokenDataset(train_df,text_vocab_len,target_vocab_len),
    'val': TokenDataset(val_df,text_vocab_len,target_vocab_len)
    }

    dataloader = {
        'train':
        torch.utils.data.DataLoader(datasets['train'],
                                batch_size=16,
                                shuffle=True,
                                num_workers=0),  
        'val':
        torch.utils.data.DataLoader(datasets['val'],
                                batch_size=16,
                                shuffle=False,
                                num_workers=0)  
    }

    model = CNN_LSTM(len(vocab), n_classes = n_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr = 3e-4)

    all_true_labels, all_preds, inputs_str, loss, acc, f1 = train_model(model, criterion, optimizer, epoch_num, dataloader,device)
    graf(loss, acc, f1)

    torch.save(model.state_dict(), 'weights/cnn_lstm.pth')
