import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from net.AlexNet import alexnet
from net.Inception import inception_v3
from net.ResNet import resnet50
import pandas as pd
from prepare import GetData
from torch.autograd import Variable
from PIL import Image
import os
from torchvision import transforms


class l_bfgs_net(nn.Module):
    def __init__(self):
        super(l_bfgs_net, self).__init__()
        self.delta = nn.Parameter(
            data=torch.zeros(1, 3, 299, 299), requires_grad=True)

    def forward(self, x):
        return torch.clamp(x + self.delta, 0, 1)


class Attack:
    def __init__(self,
                 model,
                 method='i_fgsm_momentum',
                 csvFile='data/CSVFile/dev.csv',
                 epsilon=0.05,
                 alpha=1 / 255,
                 iteration=20,
                 momentum_gamma=0.5,
                 weight=None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.iteration = iteration
        if weight is None:
            self.weight = [1 / len(model) for _ in model]
        elif len(weight) != len(model):
            raise (IOError('Length of weight is not same as length of model'))
        else:
            s = sum(weight)
            self.weight = [w / s for w in weight]
        self.model = [m.cuda() for m in model]
        for m in self.model:
            m.eval()
            for para in m.parameters():
                para.requires_grad = False
        self.dataLoader = GetData(
            csvFile, 1, num_workers=5, shuffle=False).getLoader()
        self.method = method
        self.momentum_gamma = momentum_gamma

    def __l_bfgs(self, img, label):
        net = l_bfgs_net()
        net = net.cuda()
        opt = optim.Adam([net.delta])
        cost = nn.CrossEntropyLoss()
        label = torch.argmax(label, dim=1)
        pre = self.model(img)
        gamma = 0.8
        for i in range(self.iteration):
            newImg = net(img)
            newPre = self.model(newImg)
            loss = -cost(newPre, label) * gamma + torch.mean(
                torch.pow(net.delta, 2)) * (1 - gamma)
            opt.zero_grad()
            loss.backward()
            opt.step()
        newImg = net(img)
        pre = torch.argmax(pre, dim=1)
        newPre = self.model(newImg)
        newPre = torch.argmax(newPre, dim=1)
        return newImg, label, pre, newPre

    def __i_fgsm(self, img, label):
        x = img
        label = torch.argmax(label, dim=1)
        # pre = self.model(img)
        preList = [m(img) for m in self.model]
        for i in range(self.iteration):
            # newPre = self.model(img)
            newPreList = [m(img) for m in self.model]
            loss = 0
            for j in range(len(newPreList)):
                loss += -F.cross_entropy(
                    input=newPreList[j], target=label) * self.weight[j]
            loss /= len(newPreList)
            # loss = -F.cross_entropy(input=newPre, target=label)
            # self.model.zero_grad()
            for m in self.model:
                m.zero_grad()
            if img.grad is not None:
                img.grad.data.fill_(0)
            loss.backward()
            img.grad.sign_()
            img = img - self.alpha * img.grad
            img = torch.where(img > x + self.epsilon, x + self.epsilon, img)
            img = torch.where(img < x - self.epsilon, x - self.epsilon, img)
            img = torch.clamp(img, 0, 1)
            img = Variable(img.data, requires_grad=True)
        # newPre = self.model(img)
        # newPre = torch.argmax(newPre, dim=1)
        # pre = torch.argmax(pre, dim=1)
        newPreList = [m(img) for m in self.model]
        newPreList = [torch.argmax(newPre, dim=1) for newPre in newPreList]
        preList = [torch.argmax(pre, dim=1) for pre in preList]
        return img, label, preList, newPreList

    def __i_fgsm_momentum(self, img, label):
        x = img
        label = torch.argmax(label, dim=1)
        preList = [m(img) for m in self.model]
        momentum = 0
        for i in range(self.iteration):
            newPreList = [m(img) for m in self.model]
            loss = 0
            for j in range(len(newPreList)):
                loss += -F.cross_entropy(
                    input=newPreList[j], target=label) * self.weight[j]
            loss /= len(newPreList)
            for m in self.model:
                m.zero_grad()
            if img.grad is not None:
                img.grad.data.fill_(0)
            loss.backward()
            momentum = self.momentum_gamma * momentum + img.grad
            img = img - self.alpha * torch.sign(momentum)
            img = torch.where(img > x + self.epsilon, x + self.epsilon, img)
            img = torch.where(img < x - self.epsilon, x - self.epsilon, img)
            img = torch.clamp(img, 0, 1)
            img = Variable(img.data, requires_grad=True)
        newPreList = [m(img) for m in self.model]
        newPreList = [torch.argmax(newPre, dim=1) for newPre in newPreList]
        preList = [torch.argmax(pre, dim=1) for pre in preList]
        return img, label, preList, newPreList

    def __fgsm(self, img, label):
        pre = self.model(img)
        label = torch.argmax(label, dim=1)
        loss = -F.cross_entropy(input=pre, target=label)
        self.model.zero_grad()
        if img.grad is not None:
            img.grad.data.fill_(0)
        loss.backward()
        img.grad.sign_()
        newImg = img - self.epsilon * img.grad
        newImg = torch.clamp(newImg, 0, 1)
        newPre = self.model(newImg)
        newPre = torch.argmax(newPre, dim=1)
        pre = torch.argmax(pre, dim=1)
        return newImg, label, pre, newPre

    def attack(self):
        total = [0 for _ in range(len(self.model))]
        suc = [0 for _ in range(len(self.model))]
        rig = [0 for _ in range(len(self.model))]
        path = os.path.join('data/attack/', self.method)
        transform = transforms.ToPILImage()
        if not os.path.exists(path):
            os.mkdir(path)
        for i, data in enumerate(self.dataLoader):
            img, label = data
            img = Variable(img.cuda(), requires_grad=True)
            label = label.cuda()
            if self.method == 'fgsm':
                newImg, label, pre, newPre = self.__fgsm(img, label)
            elif self.method == 'i_fgsm':
                newImg, label, preList, newPreList = self.__i_fgsm(img, label)
            elif self.method == 'i_fgsm_momentum':
                newImg, label, preList, newPreList = self.__i_fgsm_momentum(
                    img, label)
            elif self.method == 'l_bfgs':
                newImg, label, pre, newPre = self.__l_bfgs(img, label)
            else:
                raise (IOError('No method named {}'.format(self.method)))
            newImg = newImg.cpu().detach().numpy().squeeze(0).transpose((1, 2,
                                                                         0))
            img = img.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
            delta = newImg - img
            newImg = Image.fromarray(np.uint8(newImg * 255))
            img = Image.fromarray(np.uint8(img * 255))
            delta = Image.fromarray(np.uint8(delta * 255))
            newImg.save(os.path.join(path, '{:0>3}_new.jpg'.format(i)))
            img.save(os.path.join(path, '{:0>3}.jpg'.format(i)))
            delta.save(os.path.join(path, '{:0>3}_delta.jpg'.format(i)))
            flags = []
            for j in range(len(preList)):
                if preList[j] == label:
                    total[j] += 1
                    if newPreList[j] == label: rig[j] += 1
                    if newPreList[j] != preList[j]:
                        suc[j] += 1
                        flags.append('succeed')
                    else:
                        flags.append('fail')
                else:
                    flags.append('ignore')
            preStrList = [
                ' {:>3}'.format(pre.cpu().detach().numpy()[0])
                for pre in preList
            ]
            newpreStrList = [
                ' {:>3}'.format(newPre.cpu().detach().numpy()[0])
                for newPre in newPreList
            ]
            flagsStrList = ['{:>10}'.format(f) for f in flags]

            preStr = ''
            newpreStr = ''
            flagStr = ''

            for j in range(len(preList)):
                preStr += preStrList[j]
                newpreStr += newpreStrList[j]
                flagStr += flagsStrList[j]

            print('NO {:>3} :'.format(i) + 'True label is {:>3}'.format(
                label.cpu().detach().numpy()[0]) + ', prediction is' + preStr +
                  ', new prediction is' + newpreStr + '. ' + flagStr)

        accStrList = [
            ' {:.6f}'.format(rig[i] / total[i]) for i in range(len(total))
        ]
        sucStrList = [
            ' {:>3}/{:>3}'.format(suc[i], total[i]) for i in range(len(total))
        ]
        accStr = ''
        sucStr = ''
        for j in range(len(preList)):
            accStr += accStrList[j]
            sucStr += sucStrList[j]
        print('Accuracy is' + accStr)
        print('Success number is' + sucStr)


if __name__ == "__main__":
    alexnet = alexnet()
    inception = inception_v3()
    resnet = resnet50()
    alexnet.load_state_dict(torch.load('model/alexnet.pkl'))
    inception.load_state_dict(torch.load('model/inception.pkl'))
    resnet.load_state_dict(torch.load('model/resnet.pkl'))
    attack = Attack(model=[resnet, inception], weight=[1, 30])
    attack.attack()
