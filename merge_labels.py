import pandas as pd

ground_truth_train = pd.read_csv('data/old/ISBI2016_ISIC_Part3_Training_GroundTruth.csv', header=None)
ground_truth_test = pd.read_csv('data/old/ISBI2016_ISIC_Part3_Test_GroundTruth.csv', header=None)

ground_truth_train.columns = ['image','type']
ground_truth_test.columns = ['image','type']

ground_truth_test.loc[ground_truth_test['type']==1, 'type'] = 'malignant'
ground_truth_test.loc[ground_truth_test['type']==0, 'type'] = 'benign'

x = pd.concat([ground_truth_train,ground_truth_test],ignore_index=True).sort_values(
     by='image', ascending=True)

x.index = [i for i in range(len(x))]

x.to_csv('data/dataset.csv', sep='\t', encoding='utf-8')
