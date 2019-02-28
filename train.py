import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from prepare import GetData
from net.AlexNet import alexnet
from net.ResNet import resnet50
from net.Inception import inception_v3
from net.Vgg import vgg16_bn
from tqdm import tqdm
import os

useCUDA = True
# model = 'vgg'
model = 'resnet'
# model = 'alexnet'
# model = 'inception'

batch_size = 40

loader = GetData(
    'data/CSVFile/train_resized.csv', batch_size=batch_size,
    num_workers=6).getLoader()
verificationLoder = GetData(
    'data/CSVFile/dev.csv', batch_size=1, shuffle=False).getLoader()

if model == 'alexnet':
    net = alexnet()
elif model == 'resnet':
    net = resnet50()
elif model == 'inception':
    net = inception_v3()
elif model == 'vgg':
    net = vgg16_bn()

print('Using ' + model + '...')

if useCUDA:
    net = net.cuda()
    print('Using CUDA...')

if os.path.exists('model/' + model + '.pkl'):
    net.load_state_dict(torch.load('model/' + model + '.pkl'))
    print('Loading ' + model + '...')

optimizer = optim.Adam(net.parameters(), lr=1e-3)

cost = nn.CrossEntropyLoss()

for i in range(200):
    epoch_loss = 0
    total = 0
    acc = 0
    tloader = tqdm(loader)
    net.train()
    for data in tloader:
        tloader.set_description(" Training epoch {:d}".format(i))
        img, label = data
        if useCUDA:
            img = img.cuda()
            label = label.cuda()
        out = net(img)
        label = torch.argmax(label, dim=1)
        loss = cost(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data
        out = torch.argmax(out, dim=1)
        acc = (out == label).sum().cpu().numpy()
        total = img.size(0)
    print('Loss of epoch {:d} is {:.6f}'.format(i, epoch_loss))
    print('Train accuracy of epoch {:d} is {:.6f}'.format(i, acc / total))
    print('verifying...')
    total = 0
    acc = 0
    net.eval()
    for data in verificationLoder:
        img, label = data
        img = img.cuda()
        label = label.cuda()
        label = torch.argmax(label, dim=1)[0].cpu().detach().numpy()
        pre = net(img)
        pre = torch.argmax(pre, dim=1)[0].cpu().detach().numpy()
        total += 1
        if pre == label:
            acc += 1
    print('Verify accuracy of epoch {:d} is {:.6f}'.format(i, acc / total))
    print('Saving ' + model + '...')
    torch.save(net.state_dict(), 'model/' + model + '.pkl')
    print('*****************************************************')
