
import csv
from intensity_normalization.normalize.zscore import ZScoreNormalize
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.svm import SVC   
from sklearn.metrics.pairwise import chi2_kernel
import warnings
from sklearnex import patch_sklearn, config_context
patch_sklearn()
warnings.filterwarnings("ignore")



def  data_preprocessing(data_path, print_detail = True):
    data = pd.read_csv(data_path)
    # remove index
    if "Unnamed: 0" in data.columns:
        data = data.iloc[:,1:]
    # remove repetitive
    data = data.T.drop_duplicates().T
    data = data.dropna(axis=1,how='any')
    # remove useless information
    for col in data.columns:
        if col == 'id' :
            continue
        else:
            try:
                data[col] = data[col].astype(float)
            except:
                del data[col]
    if print_detail:
        print(data['rec'].value_counts())
        print("data amount: {}, feature amount: {}".format(data.shape[0],data.shape[1]))
    return data




def data_normalization(data):
    scaler = preprocessing.MinMaxScaler()
    norm_data = scaler.fit_transform(data)
    #norm_data = preprocessing.normalize(norm_data, norm='l2')
    norm_data = pd.DataFrame(norm_data,columns = data.columns)
    return norm_data





def feature_selection_rf(data, label, feature_num):
    rfc = RandomForestClassifier(n_estimators=1500)
    feature_importance = rfc.fit(data, label).feature_importances_
    # select most important features
    sort_ind = np.argsort(feature_importance,)[::-1]
    reduce_col = [data.columns[sort_ind[i]] for i in range(feature_num)]
    # add age and sex if not be selected
    
    for col in ['age','sex']:
        if col not in reduce_col:
            reduce_col.append(col)
    
    data = data.loc[:,reduce_col]

    return data




def kernel(trainvec, testvec, typeNum):
    #each row as a point
    trainPoint = trainvec.shape[0]
    testPoint = testvec.shape[0]
    dim = trainvec.shape[1]
    
    trainMatrix = np.zeros((trainPoint, trainPoint))
    testMatrix = np.zeros((testPoint, trainPoint))
    
    ## RBF kernel
    if typeNum == 1:
        for i in range(trainPoint):
            for j in range(i, trainPoint):
                trainMatrix[i,j] = np.sum((trainvec[i,:] - trainvec[j,:]) ** 2)
                trainMatrix[j,i] = trainMatrix[i,j]
        
        for i in range(testPoint):
            for j in range(trainPoint):
                testMatrix[i,j] = np.sum((testvec[i,:] - trainvec[j,:]) ** 2)
        
        gamma = np.sum(trainMatrix)
        gamma = gamma / (trainPoint ** 2)
        gamma = 1 / gamma
        trainMatrix = np.exp(-gamma * trainMatrix)
        testMatrix = np.exp(-gamma * testMatrix)
    
    ## linear
    if typeNum == 2:
        for i in range(trainPoint):
            for j in range(i, trainPoint):
                trainMatrix[i,j] = np.sum(trainvec[i,:] * trainvec[j,:])
                trainMatrix[j,i] = trainMatrix[i,j]
        
        for i in range(testPoint):
            for j in range(trainPoint):
                testMatrix[i,j] = np.sum(testvec[i,:] * trainvec[j,:])
    
    ## chi square kernel
    if typeNum == 3:
        epsilon = 1e-10
        for i in range(trainPoint):
            for j in range(i, trainPoint):
                d1 = trainvec[i,:] - trainvec[j,:]
                d2 = trainvec[i,:] + trainvec[j,:]
                d3 = (d1 ** 2) / (d2 + epsilon)
                trainMatrix[i,j] = np.sum(d3)
                trainMatrix[j,i] = trainMatrix[i,j]
        
        for i in range(testPoint):
            for j in range(trainPoint):
                d1 = testvec[i,:] - trainvec[j,:]
                d2 = testvec[i,:] + trainvec[j,:]
                d3 = (d1 ** 2) / (d2 + epsilon)
                testMatrix[i,j] = np.sum(d3)
        
        gamma = np.sum(trainMatrix)
        gamma = gamma / (trainPoint ** 2)
        gamma = 1 / gamma
        trainMatrix = np.exp(-gamma * trainMatrix)
        testMatrix = np.exp(-gamma * testMatrix)
    
    ## Histogram Intersection
    if typeNum == 4:
        for i in range(trainPoint):
            for j in range(i, trainPoint):
                HItmp = np.minimum(trainvec[i,:], trainvec[j,:])
                trainMatrix[i,j] = np.sum(HItmp)
                trainMatrix[j,i] = trainMatrix[i,j]
        
        for i in range(testPoint):
            for j in range(trainPoint):
                HItmp = np.minimum(testvec[i,:], trainvec[j,:])
                testMatrix[i,j] = np.sum(HItmp)
        
        gamma = np.sum(trainMatrix)
        gamma = gamma / (trainPoint ** 2)
        gamma = 1 / gamma
    return (gamma,trainMatrix,testMatrix)







def get_cv_score_svm(data,c):
    n_splits = 5
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2022)
    f1_test, f1_train = [],[]
    pre_test, pre_train = [],[]
    rec_test, rec_train = [],[]
    auc_test = []

    data,label = data.iloc[:,:-1], data.iloc[:,-1]
    for train_ind, test_ind in folds.split(data, label):
        x_train, y_train = data.iloc[train_ind,:], label.iloc[train_ind]
        x_test, y_test = data.iloc[test_ind,:], label.iloc[test_ind]
        if gamma != None:
            k_train = chi2_kernel(x_train,gamma=gamma)
            k_test = chi2_kernel(x_test,x_train,gamma=gamma)
        else:
            k_train = chi2_kernel(x_train)
            k_test = chi2_kernel(x_test,x_train)
        
        svm = SVC(C=c,kernel='precomputed',cache_size=200,probability=True,class_weight='balanced').fit(k_train.T,y_train)
        y_pred = svm.predict(k_test)
        y_pred_prob = svm.predict_proba(k_test)[:,1]
        y_train_pred = svm.predict(k_train)
        y_train_pred_prob = svm.predict_proba(k_train)[:,1]


        f1_test.append(f1_score(y_test, y_pred, average='micro'))
        f1_train.append(f1_score(y_train, y_train_pred, average='micro'))
        pre_test.append(precision_score(y_test, y_pred, average='micro'))
        pre_train.append(precision_score(y_train, y_train_pred, average='micro'))
        rec_test.append(recall_score(y_test, y_pred, average='micro'))
        rec_train.append(recall_score(y_train, y_train_pred, average='micro'))     
        auc_test.append(roc_auc_score(y_test, y_pred_prob))

    f = sum(f1_test) / n_splits
    p = sum(pre_test) / n_splits
    r = sum(rec_test) / n_splits
    auc = sum(auc_test) / n_splits
    return round(p,3),round(r,3),round(f,3),round(auc,3),c


def auc_select():
    for feature in features:
        st1_data, st1_label = data.iloc[:,:-1], data.iloc[:,-1]
        st1_data = st1_data[feature]
        st1_data_label = pd.concat([st1_data,st1_label],axis=1)
        aucs.append(get_cv_score_svm(st1_data_label,c)[3])
    rows = zip(features,aucs)
    with open("./all_features_auc.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)




# main
gamma = None #if set gamma, use kernel function
c = 3 #SVM's hyperparameter
feature_num = 10 
iter_num = 50
best_col = {} #chosen features and their chosen times
data = data_preprocessing("./data.csv")
feature_list = pd.read_csv("./features.csv")
features = feature_list.columns.tolist()
data = data.loc[(data['fold']=='train')]
data.drop(columns = ['fold'], inplace=True)

data = data.reset_index(drop=True)
st1_data, st1_label = data.iloc[:,:-1], data.iloc[:,-1] #last column is label
aucs = []
data_temp = data_normalization(st1_data)
data = pd.concat([data_temp,st1_label],axis=1)  

for _ in range(iter_num):
    threshold = 0.8
    st1_data, st1_label = data.iloc[:,:-1], data.iloc[:,-1]
    st1_data = feature_selection_rf(st1_data, st1_label, feature_num=feature_num) #select features
    st1_data_label = pd.concat([st1_data,st1_label],axis=1)
    output = get_cv_score_svm(st1_data_label,c) #get svm score
    print("{} attempt: {}".format(_+1,output))
    # if f1score>threshold then save
    if output[2]>threshold:
        for name in st1_data.columns.tolist():
            if name in best_col.keys():
                best_col[name]+=1
            else: 
                best_col[name] = 1
with open('./RF_select.txt', 'w') as f:
    for feature,times in best_col.items():
        f.write(str(feature)+' '+str(times)+'\n')




