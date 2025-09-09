# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import FastText

class FastTextWrapper:
    def __init__(self, sentences, embed_dim=128, window=3, epochs=10, min_count=1):
        # sentences: list of lists of "words" (we'll pass tokenized events)
        tokenized = [s.split() for s in sentences]
        self.model = FastText(vector_size=embed_dim, window=window, min_count=min_count)
        self.model.build_vocab(tokenized)
        self.model.train(tokenized, total_examples=len(tokenized), epochs=epochs)
        self.dim = embed_dim

    def embed_event(self, event_token):
        return self.model.wv[event_token]

    def embed_visit(self, visit_tokens):
        # visit_tokens: list of tokens
        vecs = [self.model.wv[t] for t in visit_tokens]
        return torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32)

    def get_vector(self, token):
        return self.model.wv[token]

# FT1-LSTM: LSTM with a time gate Tt computed from Δt and input
class FT1LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # gates
        self.Wxi = nn.Linear(input_size, hidden_size)
        self.Whi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wxf = nn.Linear(input_size, hidden_size)
        self.Whf = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wxc = nn.Linear(input_size, hidden_size)
        self.Whc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wxo = nn.Linear(input_size, hidden_size)
        self.Who = nn.Linear(hidden_size, hidden_size, bias=False)

        # time gate params (Eq.11): Tt = σn(xtWxn + σMn(Mnt Wnn) + bn)
        self.Wxn = nn.Linear(input_size, hidden_size)
        self.Wnt = nn.Linear(1, hidden_size)  # M_nt is scalar delta-time
        self.bn = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h_prev, c_prev, delta_t):
        # x: (batch, input_size)
        # delta_t: (batch,1)
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev))
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev))
        o = torch.sigmoid(self.Wxo(x) + self.Who(h_prev))
        # time gate
        t_inp = self.Wxn(x) + torch.sigmoid(self.Wnt(delta_t)) + self.bn
        Tt = torch.sigmoid(t_inp)  # shape (batch, hidden)
        # candidate
        c_tilde = torch.tanh(self.Wxc(x) + self.Whc(h_prev))
        c = f * c_prev + i * Tt * c_tilde
        h = o * torch.tanh(c)
        return h, c

class FT1LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = FT1LSTMCell(input_size, hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_seq, delta_t_seq):
        # x_seq: (seq_len, batch, input_size)
        seq_len, batch, _ = x_seq.shape
        h = torch.zeros(batch, self.hidden_size, device=x_seq.device)
        c = torch.zeros(batch, self.hidden_size, device=x_seq.device)
        for t in range(seq_len):
            x_t = x_seq[t]
            dt = delta_t_seq[t].unsqueeze(1)  # ensure (batch,1)
            h, c = self.cell(x_t, h, c, dt)
        out = self.classifier(h).squeeze(-1)
        return out
