import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # Bahdanau Attention as it allows searching info from text and calculating weight
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_a = nn.Linear(hidden_dim * 2, hidden_dim, bias=False) # *2 for bidirectional encoder
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: (batch_size, hidden_dim) - from decoder LSTM
        # encoder_outputs: (batch_size, seq_len, hidden_dim * 2)
        
        batch_size, seq_len, _ = encoder_outputs.size()
        
        hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1) # (batch_size, seq_len, hidden_dim)
        
        energy = torch.tanh(self.W_a(hidden_expanded) + self.U_a(encoder_outputs))
        scores = self.v_a(energy).squeeze(2) # (batch_size, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
            
        alphas = F.softmax(scores, dim=1) # (batch_size, seq_len)
        
        context = torch.bmm(alphas.unsqueeze(1), encoder_outputs) # (batch_size, 1, hidden_dim * 2)
        
        return context, alphas
