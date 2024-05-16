import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score
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



data = data_preprocessing("./data.csv")
data = data.loc[(data["fold"] == 'train')]
y_train,x_train = data.iloc[:,-1],data.iloc[:,:-1]
dict1 = {}


for col_name, col_data in x_train.iteritems():
    auc = roc_auc_score(y_train, col_data) 
    dict1[col_name] = auc
dict2 = pd.DataFrame(list(dict1.items()),columns=['feature','auc'])
dict2.to_csv("./auc.csv")