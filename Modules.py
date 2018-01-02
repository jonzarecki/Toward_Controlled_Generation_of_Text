import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from Constants import USE_CUDA


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size=10, n_layers=1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.Wmu = nn.Linear(hidden_size, latent_size)
        self.Wsigma = nn.Linear(hidden_size, latent_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)

    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda() if USE_CUDA else Variable(
            torch.randn(mu.size(0), mu.size(1)))
        z = mu + eps * torch.exp(log_var / 2)  # 2 for convert var to std
        return z

    def forward(self, input, train=True):
        hidden = Variable(torch.zeros(self.n_layers, input.size(0), self.hidden_size)).cuda() if USE_CUDA else Variable(
            torch.zeros(self.n_layers, input.size(0), self.hidden_size))

        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        mu = self.Wmu(hidden[-1])
        log_var = self.Wsigma(hidden[-1])
        z = self.reparametrize(mu, log_var)

        return z, mu, log_var


class Generator(nn.Module):
    def __init__(self, hidden_size, output_size, latent_size=10, code_size=2, n_layers=1):
        super(Generator, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        # self.Wz = nn.Linear(latent_size+code_size,hidden_size)
        self.Wz = nn.Linear(latent_size, hidden_size)
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, latent, code, lengths, seq_length, training=True):
        embedded = self.embedding(input)
        # embedded = self.dropout(embedded)

        # h0
        # latent_code = torch.cat((latent,code),1) # z,c
        # hidden = self.tanh(self.Wz(latent_code)).view(self.n_layers,input.size(0),-1)
        hidden = self.tanh(self.Wz(latent)).view(self.n_layers, input.size(0), -1)
        decode = []
        # Apply GRU to the output so far
        for i in range(seq_length):
            _, hidden = self.gru(embedded, hidden)
            score = self.out(hidden.view(hidden.size(0) * hidden.size(1), -1))
            softmaxed = F.log_softmax(score, dim=1)
            decode.append(softmaxed)
            _, input = torch.max(softmaxed, 1)
            embedded = self.embedding(input.unsqueeze(1))
            # embedded = self.dropout(embedded)

        scores = torch.cat(decode, 1)

        return scores.view(input.size(0) * seq_length, -1)


class Discriminator(nn.Module):

    def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout):
        super(Discriminator, self).__init__()

        V = embed_num  # num of vocab
        D = embed_dim  # dimenstion of word vector
        C = class_num  # num of class
        Ci = 1
        Co = kernel_num  # 100
        Ks = kernel_sizes  # [3,4,5]

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])


        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x, train=True):
        x = self.embed(x)  # (N,W,D)

        # if self.args.static:
        #    x = Variable(x)

        x = x.unsqueeze(1)  # (N,Ci,W,D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        if train:
            x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.fc1(x)  # (N,C)
        return logit