import torch

USE_CUDA = torch.cuda.is_available()
max_features = 10000
maxlen = 20
batch_size = 128
epoch = 1
c_dim = 2
d_word_vec = 150
lambda_c = 0.1
lambda_z = 0.1

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


HIDDEN_SIZE = 300
LATENT_SIZE = 10
CODE_SIZE = 2
BATCH_SIZE=32
STEP=500
KTA = 0.0
LEARNING_RATE=0.001
SEQ_LENGTH = 15