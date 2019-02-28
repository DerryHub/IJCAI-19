import os
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from createCSV import createCSV
from PIL import ImageFile


def resize(csvDir, csvFile, output_dir, output_folder, shape=[299, 299]):
    csvPath = os.path.join(csvDir, csvFile)
    outputDir = os.path.join(output_dir, output_folder)

    df = pd.read_csv(csvPath)
    imgList = np.array(df).tolist()

    for i, val in enumerate(tqdm(imgList)):
        img = Image.open(val[0])
        img = img.resize(shape)
        outputPath = os.path.join(outputDir, '{:0>3}'.format(val[1]))
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        filename = '{:0>3}'.format(val[1]) + '_' + '{:0>6d}.jpg'.format(i)
        outputPath = os.path.join(outputPath, filename)
        img = img.convert('RGB')
        img.save(outputPath)
    print()

    createCSV('data', 'resizedImage/train', 'data/CSVFile', 'train_resized.csv')


if __name__ == "__main__":
    # ImageFile.LOAD_TRUNCATED_IMAGES = True

    # if not os.path.exists('data/resizedImage/train'):
    #     os.mkdir('data/resizedImage/train')
    # resize('data/CSVFile', 'train_raw.csv', 'data', 'resizedImage')

    # treat dev

    root = 'data/IJCAI_2019_AAAC/IJCAI_2019_AAAC_dev'
    df = pd.read_csv(os.path.join(root, 'dev.csv'))
    imgList = np.array(df).tolist()
    filename = []
    trueLabel = []
    targetedLabel = []
    for val in tqdm(imgList):
        trueLabel += [val[1]]
        targetedLabel += [val[2]]
        n = val[0].split('.')[0]+'.jpg'
        filename += [os.path.join('data/resizedImage/dev', n)]
        img = Image.open(os.path.join(root, 'dev_data', val[0]))
        img = img.resize([299, 299])
        img = img.convert('RGB')
        if not os.path.exists('data/resizedImage/dev'):
            os.mkdir('data/resizedImage/dev')
        img.save(os.path.join('data/resizedImage/dev', n))
    d = {}
    d['filename'] = filename
    d['trueLabel'] = trueLabel
    d['targetedLabel'] = targetedLabel
    df = pd.DataFrame(d)
    df.to_csv('data/CSVFile/dev.csv', index=False)