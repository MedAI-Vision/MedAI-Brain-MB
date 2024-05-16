import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import warnings
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def data_normalization(data):
    """
    Minmax Scaler + L2 Normalization
    :param data: original feature data [dataframe]
    :return: rescaled data [dataframe]
    Only for feature data, not for label and cohort number
    """
    norm_data = data
    scaler = preprocessing.MinMaxScaler()
    norm_data = scaler.fit_transform(norm_data)
    norm_data = preprocessing.normalize(norm_data, norm='l2')
    norm_data = pd.DataFrame(norm_data, columns=data.columns)
    return norm_data


def data_normalization_apply_cohort1_to_all(data):
    """
    Minmax Scaler based on max and min of cohort1
    :param data: feature data [dataframe]
    :return: scaled feature data [dataframe]
    """
    data_base_cohort = data.loc[(data["Cohort"] == 1)]

    # The first two columns are cohort number (1, 2) and label (WNT, SHH, Group3, Group4)
    data_base_feature, data_base_ch_num = data_base_cohort.iloc[:, 2:].astype('float'), data_base_cohort.iloc[:, 0]
    data_feature, data_ch_num = data.iloc[:, 2:].astype('float'), data.iloc[:, 0]

    # scale data for every column (feature)
    columns = data_base_feature.columns.to_list()
    for column in columns:
        mx = data_base_feature[column].max()
        mn = data_base_feature[column].min()
        data_feature[column] = data_feature[column].apply(lambda x: max(0, (x - mn) / (mx - mn)))

    # concatenate cohort number, label, and scaled feature data
    data = pd.concat([data_ch_num, data.iloc[:, 1], data_feature], axis=1)
    return data


def lgb_evaluate_lgbm(max_depth, num_leaves, learning_rate, max_bin, colsample_bytree, reg_alpha, reg_lambda,
                      subsample, min_child_samples, min_child_weight, n_estimators, scale_pos_weight, seed=1,
                      X_train=None, y_train=None, X_test=None, y_test=None, X_val=None, y_val=None, multi_cls=False):
    # Set up the parameters for the LGBM classifier. The first 12 arguments are various hyperparameters.
    params = {'max_depth': int(max_depth),
              'num_leaves': int(num_leaves),
              'learning_rate': learning_rate,
              'max_bin': int(max_bin),
              'colsample_bytree': colsample_bytree,
              'random_state': seed,
              'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda,
              'subsample': subsample,
              'min_child_samples': int(min_child_samples),
              'min_child_weight': min_child_weight,
              'n_estimators': int(n_estimators),
              'scale_pos_weight': scale_pos_weight
              }

    # Initialize the LGBMClassifier and set its parameters
    modellgb = LGBMClassifier()
    modellgb.set_params(**params)

    # Fit the model using the training data
    clf_lg = modellgb.fit(X_train, y_train)

    # Predict probabilities for the training, test, and validation sets
    p_lg_train = clf_lg.predict_proba(X_train) if multi_cls else clf_lg.predict_proba(X_train)[:, 1]
    p_lg_test = clf_lg.predict_proba(X_test) if multi_cls else clf_lg.predict_proba(X_test)[:, 1]
    p_lg_val1 = clf_lg.predict_proba(X_val) if multi_cls else clf_lg.predict_proba(X_val)[:, 1]

    # Calculate the AUC (Area Under the Curve) for the training, test, and validation sets
    lg_auc_train = roc_auc_score(y_train, p_lg_train, multi_class='ovr') if multi_cls else roc_auc_score(y_train,
                                                                                                         p_lg_train)
    lg_auc_test = roc_auc_score(y_test, p_lg_test, multi_class='ovr') if multi_cls else roc_auc_score(y_test, p_lg_test)
    lg_auc_val1 = roc_auc_score(y_val, p_lg_val1, multi_class='ovr') if multi_cls else roc_auc_score(y_val, p_lg_val1)

    # Print the AUC scores for the training, test, and validation datasets
    print('************************************')
    print('Train AUC       : ', np.round(lg_auc_train, decimals=4))
    print('Test AUC       : ', np.round(lg_auc_test, decimals=4))
    print('Validation AUC : ', np.round(lg_auc_val1, decimals=4))

    # Print the parameters of the trained model
    print(modellgb.get_params())

    # Return the AUC score for the test dataset
    return np.round(lg_auc_test, decimals=4)


def lgb_evaluate_svm(C, X_train, y_train, X_test, y_test, X_val, y_val, multi_cls=False):
    # Define the hyperparameter 'C' for the Support Vector Machine (SVM) model
    params = {'C': C}

    # Initialize the SVM Classifier with the given 'C' value, probability estimation enabled
    modellgb = SVC(C=C, probability=True, cache_size=200, class_weight='balanced')

    # Fit the classifier with the training data
    clf_lg = modellgb.fit(X_train, y_train)

    # Predict probabilities on the training, test, and validation datasets
    # For multi-class classification, return probabilities for all classes
    # for binary classification, return probability for the positive class
    p_lg_train = clf_lg.predict_proba(X_train) if multi_cls else clf_lg.predict_proba(X_train)[:, 1]
    p_lg_test = clf_lg.predict_proba(X_test) if multi_cls else clf_lg.predict_proba(X_test)[:, 1]
    p_lg_val = clf_lg.predict_proba(X_val) if multi_cls else clf_lg.predict_proba(X_val)[:, 1]

    # Calculate and print the AUC (Area Under Curve) scores for the training, test, and validation sets
    lg_auc_train = roc_auc_score(y_train, p_lg_train, multi_class='ovr') if multi_cls else roc_auc_score(y_train,
                                                                                                         p_lg_train)
    lg_auc_test = roc_auc_score(y_test, p_lg_test, multi_class='ovr') if multi_cls else roc_auc_score(y_test, p_lg_test)
    lg_auc_val = roc_auc_score(y_val, p_lg_val, multi_class='ovr') if multi_cls else roc_auc_score(y_val, p_lg_val)

    # Print the AUC scores for the training, test, and validation data
    print('************************************')
    print('Train AUC       : ', np.round(lg_auc_train, decimals=4))
    print('Test AUC       : ', np.round(lg_auc_test, decimals=4))
    print('Validation AUC : ', np.round(lg_auc_val, decimals=4))

    # Print the parameters of the SVM model
    print(modellgb.get_params())

    # Return the AUC score for the test dataset
    return np.round(lg_auc_test, decimals=4)

