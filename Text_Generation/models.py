import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, word_vocab_total_words, classifier_type=None, embedding_dim=256, hidden_dim=256, num_layers=2, dropout_rate=0.5):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(word_vocab_total_words, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, word_vocab_total_words)

    def forward(self, x):
        x = x.to(self.embedding.weight.device)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        
        # print(f'Input shape: {x.shape}')
        # print(f'Embedded shape: {embedded.shape}')
        # print(f'LSTM out shape: {lstm_out.shape}')
        
        # Since the LSTM is bidirectional, we concatenate the last hidden state of the forward direction
        # and the first hidden state of the backward direction before passing it to the fully connected layer
        # For batch_first=True, the last timestep of the forward direction is lstm_out[:, -1, :hidden_dim]
        # and the first timestep of the backward direction is lstm_out[:, 0, hidden_dim:]
        forward_last = lstm_out[:, -1, :self.lstm.hidden_size]
        backward_first = lstm_out[:, 0, self.lstm.hidden_size:]
        
        # print(f'Forward last shape: {forward_last.shape}')
        # print(f'Backward first shape: {backward_first.shape}')
        
        output = self.fc(torch.cat((forward_last, backward_first), dim=1))
        # print(f'Output shape: {output.shape}')
        
        return output
