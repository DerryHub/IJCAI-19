from net.Inception import inception_v3
from net.AlexNet import alexnet
from net.ResNet import resnet50
from prepare import GetData
import torch.nn as nn
import torch
from torchvision import models

# data, label = GetData(
#     'data/CSVFile/train_resized.csv', batch_size=32).getBatch()
# print(data.size())

# net = inception_v3().cuda()
# net.eval()
# data = data.cuda()

# net = inception_v3()
# model_dict = net.state_dict()

net = models.vgg16_bn(pretrained=True)

in_ = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_, 110)

torch.save(net.state_dict(), 'model/vgg.pkl')