import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FullyConnected(torch.nn.Module):
    ''' Fully connected NN model - Most Basic'''
    def __init__(self, 
                 input_dim: int, hidden_dim: int, output_dim: int, 
                 dropout: float, num_layers: int):
        '''
        Args:
            input_dim (int):  input vector size (max sequence length).
            hidden_dim (int): hidden dimension size.
            output_dim (int): output vector size (number of classes).
            dropout (int):    dropout rate for all applicable layers.
            layers (int):     number of layers in the middle of model.
        '''
        super(FullyConnected, self).__init__() 
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x):
        x = self.dropout(self.relu(self.linear1(x)))
        for _ in range(self.num_layers):
            x = self.dropout(self.relu(self.linear2(x)))
        x = self.dropout(self.relu(self.linear3(x)))
        return x
    
class RNN(nn.Module):
    ''' Standard RNN '''
    
    def __init__(self,
                 vocab_len: int, hidden_dim: int, output_dim: int,
                 dropout: int, num_layers: int, device,
                ):
        '''
        Args:
            vocab_len (int):  dize of the vocabulary.
            hidden_dim (int): hidden dimension size.
            output_dim (int): output vector size (number of classes).
            embed_size (int): size of the embedding.
            dropout (int):    dropout rate for all applicable layers.
            num_layers (int): number of layers in the middle of model.
            device):          the device to put the hidden tensor on.
        '''
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_len = vocab_len
        self.device = device
        self.rnn = nn.RNN(input_size=vocab_len, 
                          hidden_size=hidden_dim, 
                          num_layers = num_layers, 
                          dropout = dropout,
                          batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        # x -> batch_size, sequence_length, input_size
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        x = F.one_hot(x.long(), num_classes=self.vocab_len).float()
        out, _ = self.rnn(x, h0)
        # out -> batch_size, sequence_length, hidden_dim
        out = out[:, -1, :] # Gets us just the last dim output
        out = self.linear(out)
        return out

class RNNEmbed(nn.Module):
    ''' Standard RNN with Embedding'''
    
    def __init__(self,
                 vocab_len: int, hidden_dim: int, output_dim: int, embed_size: int,
                 dropout: int, num_layers: int, device,
                ):
        '''
        Args:
            vocab_len (int):  dize of the vocabulary.
            hidden_dim (int): hidden dimension size.
            output_dim (int): output vector size (number of classes).
            embed_size (int): size of the embedding.
            dropout (int):    dropout rate for all applicable layers.
            num_layers (int): number of layers in the middle of model.
            device):          the device to put the hidden tensor on.
        '''
        super(RNNEmbed, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.embed = nn.Embedding(vocab_len, embed_size)
        self.rnn = nn.RNN(input_size=embed_size, 
                          hidden_size=hidden_dim, 
                          num_layers = num_layers, 
                          dropout = dropout,
                          batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        # x -> batch_size, sequence_length, input_size
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        x = self.embed(x.type(torch.int64))
        out, _ = self.rnn(x, h0)
        # out -> batch_size, sequence_length, hidden_dim
        out = out[:, -1, :] # Gets us just the last dim output
        out = self.linear(out)
        return out
    
class LSTM(nn.Module):
    ''' Standard LSTM '''
    
    def __init__(self,
                 vocab_len: int, hidden_dim: int, output_dim: int, embed_size: int,
                 dropout: int, num_layers: int, device,
                ):
        '''
        Args:
            vocab_len (int):  dize of the vocabulary.
            hidden_dim (int): hidden dimension size.
            output_dim (int): output vector size (number of classes).
            embed_size (int): size of the embedding.
            dropout (int):    dropout rate for all applicable layers.
            num_layers (int): number of layers in the middle of model.
            device):          the device to put the hidden tensor on.
        '''
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.embed = nn.Embedding(vocab_len, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, 
                            hidden_size=hidden_dim, 
                            num_layers = num_layers, 
                            dropout = dropout,
                            batch_first = True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        # x -> batch_size, sequence_length, input_size
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        x = self.embed(x.type(torch.int64))
        out, (_, _) = self.lstm(x, (h0, c0))
        # out -> batch_size, sequence_length, hidden_dim
        out = out[:, -1, :] # Gets us just the last dim output
        out = self.linear(out)
        return out

class BiLSTM(nn.Module):
    ''' Bi-Directional LSTM '''
    
    def __init__(self,
                 vocab_len: int, hidden_dim: int, output_dim: int, embed_size: int,
                 dropout: int, num_layers: int, device,
                ):
        '''
        Args:
            vocab_len (int):  dize of the vocabulary.
            hidden_dim (int): hidden dimension size.
            output_dim (int): output vector size (number of classes).
            embed_size (int): size of the embedding.
            dropout (int):    dropout rate for all applicable layers.
            num_layers (int): number of layers in the middle of model.
            device):          the device to put the hidden tensor on.
        '''
        super(BiLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.embed = nn.Embedding(vocab_len, embed_size)
        self.bi_lstm = nn.LSTM(input_size=embed_size, 
                            hidden_size=hidden_dim, 
                            num_layers = num_layers, 
                            dropout = dropout,
                            batch_first = True,
                            bidirectional = True)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        # x -> batch_size, sequence_length, input_size
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(self.device)
        x = self.embed(x.type(torch.int64))
        out, (_, _) = self.bi_lstm(x)
        # out -> batch_size, sequence_length, hidden_dim
        out = out[:, -1, :] # Gets us just the last dim output
        out = self.linear(out)
        return out
    
class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(hidden_dim, d_model)
        position = torch.arange(0, hidden_dim, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
    
class Transformer(nn.Module):
    ''' Transformer Encoder with Classification Head'''
    def __init__(
        self,
        vocab_len: int, embed_size: int,
        hidden_dim: int, output_dim: int, feed_forward_dim: int,
        dropout_pos: int, dropout_transformer: int, dropout_class: int, 
        num_heads: int, num_layers: int,
    ):

        super().__init__()
        self.d_model = embed_size
        assert self.d_model % num_heads == 0, "nheads must divide evenly into d_model"

        self.emb = nn.Embedding(vocab_len, embed_size)

        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            dropout=dropout_pos,
            hidden_dim=hidden_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=feed_forward_dim,
            dropout=dropout_transformer,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(self.d_model, output_dim)

    def forward(self, x):
        x = self.emb(x.type(torch.int64)) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x