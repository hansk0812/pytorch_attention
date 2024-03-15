import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, weight, num_layers=3, dropout_p=0.2, linear=True):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, 100, _weight=weight, _freeze=True)
        self.linear_flag = linear

        if self.linear_flag:
            self.linear = nn.Linear(100, hidden_size)
        
        self.activation = nn.LeakyReLU(0.03)
        self.gru = nn.GRU(hidden_size if linear else 100, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout_p)

        for param in self.gru._parameters:
            if "bias" in param:
                self.gru._parameters[param] = nn.init.zeros_(self.gru._parameters[param])
            else:
                self.gru._parameters[param] = nn.init.xavier_normal_(self.gru._parameters[param])

        if linear:
            for param in self.linear._parameters:
                if "bias" in param:
                    self.linear._parameters[param] = nn.init.zeros_(self.linear._parameters[param])
                else:
                    self.linear._parameters[param] = nn.init.xavier_normal_(self.linear._parameters[param])

    def forward(self, input):
        if self.linear_flag:
            embedded = self.dropout(self.activation(self.linear(self.embedding(input))))
        else:
            embedded = self.dropout(self.activation(self.embedding(input)))

        output, hidden = self.gru(embedded)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size*2, hidden_size)
        self.W2 = nn.Linear(hidden_size*2, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        
        for param in self.W1._parameters:
            if "bias" in param:
                self.W1._parameters[param] = nn.init.zeros_(self.W1._parameters[param])
            else:
                self.W1._parameters[param] = nn.init.xavier_normal_(self.W1._parameters[param])
        for param in self.W2._parameters:
            if "bias" in param:
                self.W2._parameters[param] = nn.init.zeros_(self.W2._parameters[param])
            else:
                self.W2._parameters[param] = nn.init.xavier_normal_(self.W2._parameters[param])
        for param in self.V._parameters:
            if "bias" in param:
                self.V._parameters[param] = nn.init.zeros_(self.V._parameters[param])
            else:
                self.V._parameters[param] = nn.init.xavier_normal_(self.V._parameters[param])

    def forward(self, query, values, mask):
         
        query = query.reshape((query.shape[0], 2, query.shape[1]//2, query.shape[-1]))
        query = query[:,:,-1,:] #query.sum(dim=2)
        query = query.reshape((-1, 1, query.shape[-1]*2))
    
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))
        scores = scores.squeeze(2).unsqueeze(1) # [B, M, 1] -> [B, 1, M]

        # Dot-Product Attention: score(s_t, h_i) = s_t^T h_i
        # Query [B, 1, D] * Values [B, D, M] -> Scores [B, 1, M]
        # scores = torch.bmm(query, values.permute(0,2,1))

        # Cosine Similarity: score(s_t, h_i) = cosine_similarity(s_t, h_i)
        # scores = F.cosine_similarity(query, values, dim=2).unsqueeze(1)

        # Attention weights
        alphas = F.softmax(scores, dim=-1)
        
        # Mask out invalid positions.
        scores.data.masked_fill_(mask.unsqueeze(1) == 1, 0) #float('inf'))

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, values)

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        return context, alphas


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, weight, dropout_p=0.2, linear=True):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, 100, _weight=weight, _freeze=True)

        self.linear_flag = linear

        if self.linear_flag:
            self.linear = nn.Linear(100, hidden_size)
            self.activation = nn.LeakyReLU(0.03)
        
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(3 * hidden_size if self.linear_flag else 2*hidden_size + 100, hidden_size, 
                            batch_first=True, num_layers=num_layers, bidirectional=True)
        self.out = nn.Linear(hidden_size*2, output_size)

        self.dropout = nn.Dropout(dropout_p)

        for param in self.gru._parameters:
            if "bias" in param:
                self.gru._parameters[param] = nn.init.zeros_(self.gru._parameters[param])
            else:
                self.gru._parameters[param] = nn.init.xavier_normal_(self.gru._parameters[param])
        
        if self.linear_flag:
            for param in self.linear._parameters:
                if "bias" in param:
                    self.linear._parameters[param] = nn.init.zeros_(self.linear._parameters[param])
                else:
                    self.linear._parameters[param] = nn.init.xavier_normal_(self.linear._parameters[param])
        
        for param in self.out._parameters:
            if "bias" in param:
                self.out._parameters[param] = nn.init.zeros_(self.out._parameters[param])
            else:
                self.out._parameters[param] = nn.init.xavier_normal_(self.out._parameters[param])

    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, SOS_token=0, max_len=10):
        # Teacher forcing if given a target_tensor, otherwise greedy.
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = torch.ones_like(encoder_hidden) # TODO: Consider bridge
        decoder_outputs = []
        
        attn_map = []

        for i in range(max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask)
            decoder_outputs.append(decoder_output)
            
            attn_map.append(attn_weights)
            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [B, Seq, OutVocab]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attn_map = torch.cat(attn_map, dim=1)
        return decoder_outputs, decoder_hidden, attn_map

    def forward_step(self, input, hidden, encoder_outputs, input_mask):
        # encoder_outputs: [B, Seq, D]
        query = hidden.permute(1,0,2)
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)
        
        if self.linear_flag:
            embedded = self.dropout(self.activation(self.linear(self.embedding(input))))
        else:
            embedded = self.dropout(self.embedding(input))
        
        attn = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(attn, hidden)
        output = self.out(output)
        # output: [B, 1, OutVocab]
        return output, hidden, attn_weights


class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size, input_vocab_size, output_vocab_size, num_layers, weights, dropout_p=0.2, linear=True, SOS_token=0):
        super(EncoderDecoder, self).__init__()

        self.SOS_token = SOS_token

        self.encoder = EncoderRNN(input_vocab_size, hidden_size, num_layers=num_layers, weight=weights[0], dropout_p=dropout_p, linear=linear)
        self.decoder = AttnDecoder(hidden_size, output_vocab_size, num_layers=num_layers, weight=weights[1], dropout_p=dropout_p, linear=linear)
        # self.decoder = DecoderRNN(hidden_size, output_vocab_size)

    def forward(self, inputs, input_mask, max_len, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs)

        decoder_outputs, decoder_hidden, attn_map = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets, 
            max_len=max_len, SOS_token=self.SOS_token)

        return decoder_outputs, decoder_hidden, attn_map
