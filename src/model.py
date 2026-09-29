import torch
import torch.nn as nn
from src.attention import Attention

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, seq_lengths):
        embedded = self.embedding(x)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: (batch_size, seq_len, hidden_dim * 2)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        
        hidden = self.fc_hidden(hidden)
        cell = self.fc_cell(cell)
        
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = Attention(hidden_dim)
        
        self.lstm = nn.LSTM(output_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_step, hidden, cell, encoder_outputs, mask=None):
        # input_step: (batch_size, 1, output_dim)
        context, alphas = self.attention(hidden, encoder_outputs, mask)
        
        rnn_input = torch.cat((input_step, context), dim=2)
        
        output, (hidden, cell) = self.lstm(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        
        prediction = self.fc_out(output.squeeze(1)) # (batch_size, output_dim)
        
        return prediction, hidden.squeeze(0), cell.squeeze(0), alphas

class Seq2SeqTTS(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(Seq2SeqTTS, self).__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim)
        self.decoder = Decoder(output_dim, hidden_dim)
        self.output_dim = output_dim

    def forward(self, text, seq_lengths, target_spectrogram=None, teacher_forcing_ratio=0.5):
        batch_size = text.shape[0]
        target_len = target_spectrogram.shape[1] if target_spectrogram is not None else 1000
            
        encoder_outputs, hidden, cell = self.encoder(text, seq_lengths)
        mask = (text != 0)
        
        outputs = torch.zeros(batch_size, target_len, self.output_dim).to(text.device)
        input_step = torch.zeros(batch_size, 1, self.output_dim).to(text.device)
        
        for t in range(target_len):
            prediction, hidden, cell, alphas = self.decoder(input_step, hidden, cell, encoder_outputs, mask)
            outputs[:, t, :] = prediction
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio if target_spectrogram is not None else False
            input_step = target_spectrogram[:, t, :].unsqueeze(1) if teacher_force else prediction.unsqueeze(1)
                
        return outputs


class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, target_len, output_dim):
        super(SimpleNN, self).__init__()
        # The input layer is the word embedding vector encoded from original sentences
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        input_size = max_seq_len * embed_dim
        self.target_len = target_len
        self.output_dim = output_dim
        
        # The hidden layer is fully connected with same number of neurons as the input layer
        self.hidden = nn.Linear(input_size, input_size)
        
        # The output layer is a stretched 1D vector of the spectrogram
        self.output = nn.Linear(input_size, target_len * output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Embed and reshape to 1D arrays to match the neural network structure
        x = self.embedding(x)
        x = x.view(batch_size, -1)
        
        x = torch.relu(self.hidden(x))
        
        # Output layer uses sigmoid activation function to squish results (0 to 1)
        x = torch.sigmoid(self.output(x))
        
        # Reshape the prediction vectors back to matrices for post-processing
        return x.view(batch_size, self.target_len, self.output_dim)

