import pandas as pd
import os
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time
from uncertainties import unumpy
import itertools

from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#from plot_cm import plot_confusion_matrix
from matplotlib.pyplot import cm

def KS_test(X_train,benign_train,malignant_train,return_maps=False,params='DEFAULT'):
    '''
    returns relevant maps from the non-parametric KS-test
    '''
    benign = X_train[benign_train]
    malignant = X_train[malignant_train]
    ks_all = np.zeros((X_train.shape[1],2))
    for index in range((X_train.shape[1])):
        ks_all[index] = st.ks_2samp(benign[:,index],
                                    malignant[:,index])

    maps =  pd.DataFrame(ks_all.T,columns=
                             ['map_'+str(i) for i in range(X_train.shape[1])],
                             index=['statistic','p-value']).T.sort_values(
                             by='statistic', ascending=False)
    if params=='DEFAULT':
        nop = min(X_train.shape) # minimum number of maps for PCA
    else:
        nop = params

    map_numbers = [int(i.split('_')[1]) for i in maps.index][:nop]


    return X_train[:,map_numbers], map_numbers #parameter numbers of 1x4096


def balanced_subsample(x,y,subsample_size=1.0):
    '''
    args, X-data, y-labels (1D array), subsample_size=1.0 --> 50/50 split
    returns a balanced dataset
    '''
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

def my_SVM(X_train, X_test,y_train, y_test,p1,p2):
    ''' SVM implementation to be used in CV
    p1 - C
    p2 - gamma
    '''
    clf = SVC(probability=True, C=p1,gamma=p2)
    clf.fit(X_train, y_train)

    y_predict=clf.predict(X_test)
    #q = np.unique(y_predict, return_counts=True)
    #print(q)
    acc = accuracy_score(y_test, y_predict)
    pre = precision_score(y_test, y_predict)
    rec = recall_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    x = {"accuracy":acc,
            "recall":rec,
            "precision":pre,
            "f1-score":f1}

    y_score = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
    roc = [fpr, tpr]
    return pd.DataFrame(list(x.values()),index=list(x.keys()),columns=['model']).T, roc

def scale_set(train,test):
    """uses sklearn standard sclar to normalize data"""
    sc = StandardScaler()
    fitted = sc.fit(train)
    return sc.transform(train), sc.transform(test)

def my_PCA(train,test):
    '''applies PCA to the data'''
    pca = PCA(n_components=train.shape[1])
    data = pca.fit(train)
    return pca.transform(train), pca.transform(test)

def classifier_1(X_train,X_test,benign_train,malignant_train):
    '''
    Ks-test(minimum) --> PCA --> Ks-test(256)
    '''
    X_train, numbers = KS_test(X_train,benign_train,malignant_train,
                                                      return_maps=True)
    X_test = np.array(pd.DataFrame(X_test.T).iloc[numbers].T)

    ## PCA
    X_train,X_test = my_PCA(X_train,X_test)

    ## second KS-test
    X_train, numbers = KS_test(X_train,benign_train,malignant_train,
                          return_maps=True,params=256)
    X_test = np.array(pd.DataFrame(X_test.T).iloc[numbers].T)
    return X_train,X_test

def classifier_2(X_train,X_test,benign_train,malignant_train):
    '''
    Ks-test(256)
    '''
    X_train, numbers = KS_test(X_train,benign_train,malignant_train,
                          return_maps=True,params=256)
    X_test = np.array(pd.DataFrame(X_test.T).iloc[numbers].T)
    return X_train,X_test, numbers

def model(X,y, balanced = False,n_splits=5, DR = 2, seed=10, model_name='classic'):
    '''
    Function to get metrics of my KS/PCA/KS classifier

    Args: X-data, y-labels,
    balanced - if True, makes dataset balanced
    n_splits - K-fold cross validation splits
    DR - Dimensionality reduction type:
    1: ks-test, pca, ks-test
    2: only ks-test
    else: None

    '''

    if balanced:
        X,y = balanced_subsample(X,y)
    if int(DR[-1]) == 0:
        C = np.logspace(1,3,10)[2:5]
        gamma = [1e-5]
    else:
        C = np.logspace(1,3,10)
        gamma = [1e-5,1e-6]
    #C = [129.2]
    #gamma = [1e-5]
    #C = [46,129]
    #gamma = [1e-5]
    permutations = [(x,y) for x in C for y in gamma]

    kf = StratifiedKFold(n_splits=n_splits,random_state=42, shuffle=True)
    np.random.seed(seed)
    baseline = pd.DataFrame(np.zeros(4)).T
    baseline.columns = ['accuracy','recall','precision','f1-score']

    for p1,p2 in permutations:
        print(p1,p2)
        empty = pd.DataFrame([],columns=['accuracy','recall','precision','f1-score'])

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for train_index, test_index in kf.split(X,y):


            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            benign_train = np.where(y_train==0)[0]
            benign_test = np.where(y_test==0)[0]
            malignant_train = np.where(y_train==1)[0]
            malignant_test = np.where(y_test==1)[0]
            X_train, X_test = scale_set(X_train,X_test)

            if int(DR[-1]) == 1: #kstest,pca, ks-test
                X_train, X_test = classifier_1(X_train,X_test,benign_train,malignant_train)
            elif int(DR[-1]) == 2: #only ks test
                X_train, X_test, numbers = classifier_2(X_train,X_test,benign_train,malignant_train)
            else:
                pass
            #metrics calcs
            performance, roc = my_SVM(X_train, X_test, y_train, y_test,p1,p2)
            empty = pd.concat([empty,performance])

            #roc curve calcs
            tprs.append(scipy.interp(mean_fpr, roc[0], roc[1]))
            tprs[-1][0] = 0.0
            roc_auc = auc(roc[0], roc[1])
            aucs.append(roc_auc)

        permuation_performance = empty.mean() #mean of scores for all CVs
        if np.array([permuation_performance['f1-score']])>=np.array([baseline['f1-score']]): #maximizing f1 score
            baseline = pd.DataFrame(unumpy.uarray(permuation_performance, empty.std())).T

            baseline.columns = ['accuracy','recall','precision','f1-score']
            baseline['$C$'],baseline['$\gamma$'] = p1, p2
            tprs_best = tprs
            aucs_best = aucs
            mean_fpr_best = mean_fpr

    r = {"tprs":tprs_best,
         "aucs":aucs_best,
         "mean_fpr":mean_fpr_best} #roc

    #returning part
    baseline.insert(0, 'Model', model_name)
    if int(DR[-1]) == 1:
        baseline.insert(1, 'Dimentionality Reduction', DR)
    elif int(DR[-1]) == 2:
        baseline.insert(1, 'Dimentionality Reduction', DR)
    else:
        baseline.insert(1, 'Dimentionality Reduction', DR)

    return baseline,r,numbers


plt.rcParams['text.usetex']=True
plt.rcParams['text.latex.unicode']=True
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

def plot_roc2(rocs,DR,model_name,balanced=True):
    '''plots the roc curve for a given model'''
    plt.figure(figsize=(10,7))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)
    #colors = ['b','k','r']
    for index, values in enumerate(rocs):
        tprs = values['tprs']
        aucs = values['aucs']
        mean_fpr = values['mean_fpr']

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr,
                 label='{}-{} (AUC = {:.2f} $\pm$ {:.2f})'.format(DR[index],model_name,mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.2)


    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic of the {} model'.format(mode_name))
    plt.legend(loc="lower right")
    if balanced:
        plt.savefig("thesis_text/Pictures/balanced_"+model_name+".png",bbox_inches='tight')
    else:
        plt.savefig("thesis_text/Pictures/imbalanced_"+model_name+".png",bbox_inches='tight')
    plt.show()

def plot_roc(rocs,DR,mode_name,balanced=True):
    '''plots the roc curve for a given model'''
    plt.figure(figsize=(10,7))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='cyan',
         label='Luck', alpha=.8)
    colors = ['b','k','r']
    for index, values in enumerate(rocs):
        tprs = values['tprs']
        aucs = values['aucs']
        mean_fpr = values['mean_fpr']

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr,
                 label='{} (AUC = {:.2f} $\pm$ {:.2f})'.format(DR[index],mean_auc, std_auc),
                 lw=2, alpha=.8,color=colors[index])

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.2,color=colors[index])


    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic of the {} model'.format(mode_name))
    plt.legend(loc="lower right")
    if balanced:
        plt.savefig("thesis_text/Pictures/balanced_"+mode_name+".png",bbox_inches='tight')
    else:
        plt.savefig("thesis_text/Pictures/imbalanced_"+mode_name+".png",bbox_inches='tight')
    #plt.show()

def save_scores(f,balanced=True):
    '''saves scores in seperate tables for each model'''
    if balanced:
        type_='balanced'
    else:
        type_='imbalanced'
    path = 'final_results/'
    f.T.to_csv(path+type_+'all.csv', sep='\t', encoding='utf-8',header=False)
    f = f.drop('Model',1)

    f[:3].T.to_csv(path+type_+'_classic.csv', sep='\t', encoding='utf-8',header=False)
    f[3:6].T.to_csv(path+type_+'_non_classic_ave.csv', sep='\t', encoding='utf-8',header=False)
    f[6:9].T.to_csv(path+type_+'_non_classic_max.csv', sep='\t', encoding='utf-8',header=False)


def train(data, y, balanced, models, DR, permutations):
    '''trains the full pipeline'''
    start = time.time()
    scores = []
    rocs = []
    for index, (mod, c1) in enumerate(permutations):
        s_time = time.time()
        print("-"*20)
        print("permutation {}/9, {} model, {}".format(index+1,mod,c1))
        score, r = model(X = np.array(data[mod].T),
                        y = y,
                        balanced = balanced,
                        n_splits = 5,
                        DR = c1,
                        seed = 42,
                        model_name = mod)
        print(score)
        end = time.time()
        total = end-start
        current = end-s_time
        print("Permutation executed in {:.2f}m\nTotal time eclipsed: {:.2f}m\n".format(current/60,total/60))
        scores.append(score)
        rocs.append(r)
        if index in [2,5,8]:
            plot_roc(rocs,
                     DR,
                     mod,
                     balanced = balanced)
            rocs=[]
    f = pd.concat(scores, axis=0)
    save_scores(f,balanced = balanced)
