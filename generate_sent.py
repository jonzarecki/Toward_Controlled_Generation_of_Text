import cPickle as pkl
from torch.autograd import Variable

from Constants import *
from Modules import Encoder, Generator

word2index = pkl.load(open('models/word2index.pkl', 'r'))
index2word = pkl.load(open('models/index2word.pkl', 'r'))

encoder = Encoder(len(word2index), HIDDEN_SIZE, LATENT_SIZE, 2)
generator = Generator(HIDDEN_SIZE, len(word2index), LATENT_SIZE, CODE_SIZE)
generator.load_state_dict(pkl.load(open('models/generator.pkl', 'r')))
encoder.load_state_dict(pkl.load(open('models/encoder.pkl', 'r')))

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
