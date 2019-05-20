import torch
from torch import nn
from torch.nn import init
from utils import reverse_padded_sequence


class BiRNN(nn.Module):
    '''
    Class nhận input là các word vector, sau đó encode để sinh ra sentence vector
    '''

    def __init__(self, rnn_type="GRU", **kwargs):
        super().__init__()
        # input của nn.GRU có size là (seq_len, batch, input_size)

        self.rnn_type = rnn_type
        if rnn_type == "GRU":
            rnn = nn.GRU
        elif rnn_type == "LSTM":
            rnn = nn.LSTM
        else:
            raise "Rnn type {} is not valid".format(rnn_type)

        self.forward_encoder = rnn(input_size=kwargs['word_dim'],        # word_dim = 300 (or 100)
                                      hidden_size=kwargs['hidden_dim'],
                                      num_layers=kwargs['num_layers'],
                                      # dropout=kwargs['dropout'])
                                      dropout=0)
        self.backward_encoder = rnn(input_size=kwargs['word_dim'],
                                       hidden_size=kwargs['hidden_dim'],
                                       num_layers=kwargs['num_layers'],
                                       dropout=0)
        self.reset_parameters()

    def reset_parameters(self):
        # Khởi tạo tham số
        for name, param in self.forward_encoder.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.normal_(param, mean=0, std=0.01)
        for name, param in self.backward_encoder.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.normal_(param, mean=0, std=0.01)

    def encode(self, input, length):
        # input: [bsz, len, w_dim]
        # length: [bsz, ], mỗi phần tử chỉ ra chiều dài thực sự (length[i] <= input.size[1]) của batch tương ứng
        input = torch.transpose(input, 0, 1).contiguous()  # [len, bsz, w_dim]
        reversed_input = reverse_padded_sequence(input, length, batch_first=False)

        # Encoder GRU trả về 2 đối tương là (output, hidden_state)
        forward_output = self.forward_encoder(input)[0]
        reversed_backward_output = self.backward_encoder(reversed_input)[0]  # [len, bsz, hid_dim]
        return forward_output, reversed_backward_output

    def forward(self, input, length):
        # input: [bsz, len, w_dim]
        # length: [bsz, ]
        forward_output, reversed_backward_output = self.encode(input, length)
        backward_output = reverse_padded_sequence(reversed_backward_output, length, batch_first=False)
        output = torch.cat([forward_output, backward_output], dim=2)  # [len, bsz, 2*hid_dim]
        return output


class BiRNN_wrapper(BiRNN):

    def __init__(self, rnn_type="GRU", **kwargs):
        super().__init__(rnn_type, **kwargs)

    def forward(self, input, length):
        # input.size = (batch_size, seq_len, input_dim)
        # forward_output.size = (seq_len, batch_size, hidden_dim)
        forward_output, reversed_backward_output = super().encode(input, length)
        bsz = forward_output.size(1)

        # output = []
        # for i in range(bsz):
        #     output.append(torch.cat([
        #         forward_output[length[i] - 1, i],  # forward
        #         reversed_backward_output[length[i] - 1, i]  # backward
        #     ]))
        # output = torch.stack(output)

        # Chỉ giữ output tương ứng với time_step cuối cùng
        output = torch.stack([
            torch.cat([
                forward_output[length[i] - 1, i],  # forward
                reversed_backward_output[length[i] - 1, i]  # backward
            ])  # concat the forward embedding and the backward embedding
            for i in range(bsz)])

        return output  # [bsz, 2*h_dim]


class RNN_wrapper(nn.Module):

    def __init__(self, rnn_type="GRU", **kwargs):
        super().__init__()

        self.rnn_type = rnn_type
        if rnn_type == "GRU":
            rnn = nn.GRU
        elif rnn_type == "LSTM":
            rnn = nn.LSTM
        else:
            raise "Rnn type {} is not valid".format(rnn_type)

        self.encoder = rnn(input_size=kwargs['word_dim'],
                              hidden_size=kwargs['hidden_dim'],
                              num_layers=kwargs['num_layers'],
                              dropout=kwargs['dropout'])
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                # init.kaiming_normal_(param)
                init.normal_(param, mean=0, std=0.01)

    def forward(self, input, length):
        # input: [bsz, len, w_dim]
        # length: [bsz, ]
        input = torch.transpose(input, 0, 1).contiguous()  # [len, bsz, w_dim]
        all_h = self.encoder(input)[0]
        bsz = length.size(0)
        output = torch.stack(
            [all_h[length[i] - 1, i] for i in range(bsz)]
        )
        return output  # [bsz, h_dim]


class AVG_wrapper(nn.Module):
    '''
    Tính trung bình các vector trong cùng 1 batch => Output các vector đại diện cho các batch
    '''

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input, length):
        # input: [bsz, len, w_dim]
        # length: [bsz, ]
        bsz = length.size(0)
        output = torch.stack(
            [torch.mean(input[i, :length[i]], dim=0) for i in range(bsz)]
        )
        return output  # [bsz, w_dim]


class SentenceEmbedding(nn.Module):

    def __init__(self, **kwargs):
        super(SentenceEmbedding, self).__init__()
        self.drop = nn.Dropout(0)
        self.word_embedding = nn.Embedding(num_embeddings=kwargs['num_words'],
                                           embedding_dim=kwargs['word_dim'])
        self.SE_type = kwargs['SE_type']
        if self.SE_type == 'GRU':
            self.encoder = RNN_wrapper(rnn_type="GRU", **kwargs)
            self.dim = kwargs['hidden_dim']
        elif self.SE_type == 'BiGRU':
            self.encoder = BiRNN_wrapper(rnn_type="GRU", **kwargs)
            self.dim = 2 * kwargs['hidden_dim']

        elif self.SE_type == 'LSTM':
            self.encoder = RNN_wrapper(rnn_type="LSTM", **kwargs)
            self.dim = kwargs['hidden_dim']
        elif self.SE_type == 'BiLSTM':
            self.encoder = BiRNN_wrapper(rnn_type="LSTM", **kwargs)
            self.dim = 2 * kwargs['hidden_dim']

        elif self.SE_type == 'AVG':
            self.encoder = AVG_wrapper(**kwargs)
            self.dim = kwargs['word_dim']

        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.word_embedding.weight, mean=0, std=0.01)

    def getDim(self):
        return self.dim

    def forward(self, input, length):
        # input: [bsz, len]
        # length: [bsz, ]
        input = self.drop(self.word_embedding(input))  # [bsz, len, word_dim]
        output = self.encoder(input, length)
        return output  # [bsz, self.dim]
