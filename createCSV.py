import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def createCSV(input_dir, input_folder, output_dir, output_file):
    inputPath = os.path.join(input_dir, input_folder)
    outputPath = os.path.join(output_dir, output_file)

    imgFolder = os.listdir(inputPath)
    classList = []
    imgList = []
    for path in tqdm(imgFolder):
        l = os.listdir(os.path.join(inputPath, path))
        imgList += [os.path.join(inputPath, path, val) for val in l]
        classList += [int(path) for val in range(len(l))]

    d = {}
    d['imagePath'] = imgList
    d['class'] = classList

    df = pd.DataFrame(d)
    df.to_csv(outputPath, index=False)
    

if __name__ == "__main__":
    if not os.path.exists('data/CSVFile'):
        os.mkdir('data/CSVFile')
    createCSV('data', 'IJCAI_2019_AAAC/IJCAI_2019_AAAC_train', 'data/CSVFile', 'train_raw.csv')