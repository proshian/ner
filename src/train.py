import tqdm
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import Adam, AdamW
from model import CNN_LSTM
from torch import nn

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


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_classes = len(vocab_lables)
    model = CNN_LSTM(len(vocab), n_classes = n_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr = 3e-4)
