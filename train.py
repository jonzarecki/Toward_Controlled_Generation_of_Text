import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import Counter
import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb
import torch
from torch.autograd import Variable
from torch.optim import Adam

from Constants import *
from Modules import Encoder, Generator, Discriminator

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features,
    start_char=BOS,
    oov_char=UNK,
    index_from=EOS,
)

forward_dict = imdb.get_word_index()
for key, value in forward_dict.items():
    forward_dict[key] = value + EOS
forward_dict[PAD_WORD] = PAD
forward_dict[UNK_WORD] = UNK
forward_dict[BOS_WORD] = BOS
forward_dict[EOS_WORD] = EOS

backward_dict = {}
for key, value in forward_dict.items():
    backward_dict[value] = key


def vector_to_sent(sent_vec):
    return " ".join(map(lambda idx: backward_dict[idx], sent_vec))


print("Finished Loading.")
data = [[vector_to_sent(inst[0]), inst[1]] for inst in zip(x_train, y_train)] \
       + [[vector_to_sent(inst[0]), inst[1]] for inst in zip(x_test, y_test)]

# positive = [d for d in data if d[1] == 1]
# negative = [d for d in data if d[1] == 0]
# data = random.sample(positive, 1000) + random.sample(negative, 1000)

SEQ_LENGTH = 15
train = []
for t in data:
    t0 = t[0]
    t0 = t0.replace("<br>", "")
    t0 = t0.replace("/", "")

    token0 = t0.split()

    if len(token0) >= SEQ_LENGTH:
        token0 = token0[:SEQ_LENGTH - 1]
    token0.append("<EOS>")

    while len(token0) < SEQ_LENGTH:
        token0.append('<PAD>')

    train.append([token0, token0, t[1]])

word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

for t in train:
    for token in t[0]:
        if token not in word2index:
            word2index[token] = len(word2index)

index2word = {v: k for k, v in word2index.items()}


def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], seq))
    tensor = Variable(torch.LongTensor(idxs)).cuda() if USE_CUDA else Variable(torch.LongTensor(idxs))
    return tensor


flatten = lambda l: [item for sublist in l for item in sublist]

train_x = []
train_y = []
code_labels = []
lengths = []
for tr in train:
    temp = prepare_sequence(tr[0], word2index)
    temp = temp.view(1, -1)
    train_x.append(temp)

    temp2 = prepare_sequence(tr[1], word2index)
    temp2 = temp2.view(1, -1)
    train_y.append(temp2)

    length = [t for t in tr[1] if t != '<PAD>']
    lengths.append(len(length))
    code_labels.append(
        Variable(torch.LongTensor([int(tr[2])])).cuda() if USE_CUDA else Variable(torch.LongTensor([int(tr[2])])))

train_data = list(zip(train_x, train_y, code_labels))


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        x, y, c = zip(*batch)
        x, y, c = torch.cat(x), torch.cat(y), torch.cat(c)
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp

        yield (x, y, c)


print "encoders"
encoder = Encoder(len(word2index), HIDDEN_SIZE, LATENT_SIZE, 2)
generator = Generator(HIDDEN_SIZE, len(word2index), LATENT_SIZE, CODE_SIZE)
discriminator = Discriminator(len(word2index), 100, 2, 30, [3, 4, 5], 0.8)
if USE_CUDA:
    encoder = encoder.cuda()
    generator = generator.cuda()
    discriminator = discriminator.cuda()

Recon = nn.CrossEntropyLoss(ignore_index=0)

enc_optim = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
gen_optim = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
dis_optiom = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

print "train"
for step in range(STEP):
    for i, (x, y, c) in enumerate(getBatch(BATCH_SIZE, train_data)):

        encoder.zero_grad()
        generator.zero_grad()

        generator_input = Variable(torch.LongTensor([[word2index['<SOS>']] * BATCH_SIZE])).transpose(1, 0)

        if USE_CUDA:
            generator_input = generator_input.cuda()

        latent, mu, log_var = encoder(x)

        code = Variable(torch.randn([BATCH_SIZE, 2]).uniform_(0, 1)).cuda() if USE_CUDA else Variable(
            torch.randn([BATCH_SIZE, 2]).uniform_(0, 1))

        score = generator(generator_input, latent, code, lengths, SEQ_LENGTH)
        recon_loss = Recon(score, y.view(-1))
        kld_loss = torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))

        #     KL_COST_ANNEALING
        cost_annealing_check = recon_loss.data.cpu().numpy()[0] if USE_CUDA else recon_loss.data.numpy()[0]
        if cost_annealing_check < 1.5:
            KTA = 0.5  # KL cost term annealing
        elif cost_annealing_check < 1.0:
            KTA = 0.75
        elif cost_annealing_check < 0.5:
            KTA = 1.0
        else:
            KTA = 0.0

        ELBO = recon_loss + KTA * kld_loss

        ELBO.backward()

        torch.nn.utils.clip_grad_norm(encoder.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm(generator.parameters(), 5.0)

        gen_optim.step()
        enc_optim.step()

    # KL term Anealing
    # KTA+=1/STEP
    # KTA = round(KTA,3)

    elbo_for_print = ELBO.data.cpu().numpy()[0] if USE_CUDA else ELBO.data.numpy()[0]
    recon_for_print = recon_loss.data.cpu().numpy()[0] if USE_CUDA else recon_loss.data.numpy()[0]
    kld_for_print = kld_loss.data.cpu().numpy()[0] if USE_CUDA else kld_loss.data.numpy()[0]
    print("[%d/%d] ELBO : %.4f , RECON : %.4f & KLD : %.4f" % (step, STEP, elbo_for_print,
                                                               recon_for_print,
                                                               kld_for_print))

torch.save(generator.state_dict(), 'models/generator.pkl')
torch.save(encoder.state_dict(), 'models/encoder.pkl')

# encoder = Encoder(len(word2index), HIDDEN_SIZE, LATENT_SIZE, 2)
# generator = Generator(HIDDEN_SIZE, len(word2index), LATENT_SIZE, CODE_SIZE)
# generator.load_state_dict('models/generator.pkl')
# encoder.load_state_dict('models/encoder.pkl')

generator_input = Variable(torch.LongTensor([[word2index['<SOS>']] * 1])).transpose(1, 0)
if USE_CUDA:
    generator_input = generator_input.cuda()

latent = Variable(torch.randn([1, 10])).cuda() if USE_CUDA else Variable(torch.randn([1, 10]))
code = Variable(torch.randn([1, 2]).uniform_(0, 1)).cuda() if USE_CUDA else Variable(torch.randn([1, 2]).uniform_(0, 1))
recon = generator(generator_input, latent, code, 15, SEQ_LENGTH, False)

v, i = torch.max(recon, 1)

decoded = []
for t in range(i.size()[0]):
    decoded.append(index2word[i.data.cpu().numpy()[t] if USE_CUDA else i.data.cpu().numpy()[t]])

print('A: ', ' '.join([i for i in decoded if i != '<PAD>' and i != '<EOS>']) + '\n')
