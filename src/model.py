from torch import nn

class CNN_LSTM(nn.Module):
    def __init__(self, vocab_size, n_classes, embedding_dim=250, hidden_size = 32, filters=((2, 10), (3, 8))):
        super().__init__()
        
        self.embeddings_layer = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=200, kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool1d(2)
        input_size = 100
        self.hidden_size = hidden_size
        self.lstm_layer = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, n_classes)
        

    def forward(self, inputs):
        projections = self.embeddings_layer.forward(inputs) 
        projections = projections.transpose(1, 2)
        projections = self.conv1(projections)
        projections = projections.transpose(1, 2)
        projections = self.pool1(projections)
        output, (final_hidden_state, final_cell_state) = self.lstm_layer(projections)
        output = output.reshape(-1, self.hidden_size)
        output = self.fc(output)
        return output
