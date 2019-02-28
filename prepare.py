import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np

__all__ = ['GetData']


class MyDataset(Dataset):
    def __init__(self,
                 csvFile,
                 root_dir='',
                 classNum=110,
                 transform=transforms.Compose([transforms.ToTensor()])):
        self.df = pd.read_csv(csvFile)
        self.items = self.df.iloc
        self.root_dir = root_dir
        self.transform = transform
        self.classNum = classNum

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        imgPath = self.items[index, 0]
        imgPath = os.path.join(self.root_dir, imgPath)
        classIndex = self.items[index, 1]
        img = Image.open(imgPath)
        if self.transform:
            imgTensor = self.transform(img)
        classTensor = torch.zeros(self.classNum)
        classTensor[classIndex] = 1
        return imgTensor, classTensor


class GetData(object):
    def __init__(self,
                 csvFile,
                 batch_size,
                 num_workers=0,
                 shuffle=True,
                 pin_memory=False,
                 root_dir='',
                 classNum=110,
                 transform=transforms.Compose([transforms.ToTensor()])):
        dataset = MyDataset(
            csvFile, root_dir=root_dir, classNum=classNum, transform=transform)
        self.dataloder = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory)

    def getBatch(self):
        return next(iter(self.dataloder))

    def getLoader(self):
        return self.dataloder


if __name__ == "__main__":
    # dataset = MyDataset('Data/CSVFile/resizedImages.csv')
    dataloader = GetData('data/CSVFile/dev.csv', batch_size=500).getLoader()
    for data in dataloader:
        img, label = data
        print(img.size())
        print(label)
        break