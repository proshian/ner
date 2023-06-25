from sklearn.metrics import f1_score, accuracy_score
import torch
import tqdm

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

# if __name__ == "__main__":
    # загружаем словарь, загружаем данные 
