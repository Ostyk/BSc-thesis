import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from final import train

data = {}
for i in os.listdir('data'):
    if i[-3:] == 'csv':
        name = i.split('.')[0]
        data.update({name:pd.read_csv('data/'+name+'.csv',sep='\t').iloc[:,1:]})

ground_truth = data['dataset']
labels=[]
for i in range(ground_truth.shape[0]):
    if ground_truth['type'][i] == 'benign':
        labels.append(0)
    elif ground_truth['type'][i] == 'malignant':
        labels.append(1)
y=np.array(labels)

models = ['Classic','Non-classic-ave','Non-classic-max']
DR = ['DR0','DR1','DR2']
permutations = [(x,z) for x in models for z in DR]

if __name__ == '__main__':
    from sys import argv
    b = str(argv[1])
    if b=='balanced':
        train(data, y, True, models, DR, permutations)
    elif b=='imbalanced':
        train(data, y, False, models, DR, permutations)
    elif b=='both':
        print("balanced")
        train(data, y, True, models, DR, permutations)
        print("imbalanced")
        train(data, y, False, models, DR, permutations)
