import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def feature_selection_rf(data, label, feature_num=200):
    """
    use random forest to select feature
    :param data:  data [dataframe]
    :param label: label column
    :param feature_num: the number of selected feature
    :return: selected data [dataframe]
    """
    rfc = RandomForestClassifier(n_estimators=1500)
    feature_importance = rfc.fit(data, label).feature_importances_
    sort_ind = np.argsort(feature_importance, )[::-1]
    reduce_col = [data.columns[sort_ind[i]] for i in range(feature_num)]
    sel_data = data.loc[:, reduce_col]
    return sel_data


def feature_selection_auc(data, label, range_auc=None, multi_cls=False):
    """
    use AUC scores to select feature
    :param data: data [dataframe]
    :param label: label column
    :param range_auc: [lower, upper], we think the feature with AUC score out of this range is useful
    :return: selected data [dataframe]
    """
    if range_auc is None:
        range_auc = [0.42, 0.58]
    sel_col = []
    label = label.astype(int)
    for index, row in tqdm(data.items()):
        if multi_cls:
            auc = roc_auc_score(label.tolist(), data[index].tolist(), multi_class='ovr')
        else:
            auc = roc_auc_score(label.tolist(), data[index].tolist())

        if auc < range_auc[0] or auc > range_auc[1]:
            sel_col.append(index)
    sel_data = data.loc[:, sel_col]
    return sel_data
