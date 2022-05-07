import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.preprocessing import OneHotEncoder


def label_encode(df):
    """

    :param df:
    :return: encode descriptive labels as numerical
    """
    
    label_encoder = preprocessing.LabelEncoder()
    df['diet'] = label_encoder.fit_transform(df['diet'])
    for num in "1234":
        df[f'MachineryVar{num}'] = label_encoder.fit_transform(df[f'MachineryVar{num}'])
    return df


def one_hot_encode(df, col_list=["diet", "MachineryVar1", "MachineryVar2", "MachineryVar3", "MachineryVar4"]):
    encoder = OneHotEncoder()
    encoder_df = pd.DataFrame(encoder.fit_transform(df[col_list]).toarray())
    final_df = df.join(encoder_df)

    final_df.drop(col_list, axis=1, inplace=True)
    return final_df


def read_data(path, encode):
    """

    :param path:
    :return:
    """
    
    df = pd.read_csv(path)
    if encode == "label":
        df = label_encode(df)
    if encode == "onehot":
        df = one_hot_encode(df)
    df.pop("unique_id")
    X0 = df[df['nEvent'] == 0]
    X1 = df[df['nEvent'] == 1]
    y0 = X0.pop('nEvent')
    y1 = X1.pop('nEvent')


    return X0, X1, y0, y1


def preprocess_na(df, method):
    """
    fill na with 0 or the mean of
    :param df:
    :param method:
    :return:
    """
    
    if method == "zero":
        df = df.fillna(0)
    if method == "mean":
        df = df.fillna(df.mean())
    return df


def training_data(scale, encode, na):
    """
    Prepare training data
    :param scale: if True, normalize the data
    :param encode: if "label", use label encoding; if "onehot", use one hot encoding
    :param na: if "zero", fill na with 0; if "mean", fill na with column mean
    :return: training set and test set
    """
    
    path = "../data/data.csv"
    X0, X1, y0, y1 = read_data(path, encode)
    
    # getting train/test split randomly from each class
    X_train0, X_test0, y_train0, y_test0 = train_test_split(X0, y0, test_size=0.2, random_state=42)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
    
    X_train = pd.concat([X_train0, X_train1])
    y_train = pd.concat([y_train0, y_train1])
    
    X_test = pd.concat([X_test0, X_test1])
    y_test = pd.concat([y_test0, y_test1])
    
    X_train = preprocess_na(X_train, method=na)
    X_test = preprocess_na(X_test, method=na)
    
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    y_train = np.ravel(y_train.to_numpy().reshape(-1, 1))
    y_test = np.ravel(y_test.to_numpy().reshape(-1,1))
    
    ros = RandomOverSampler(random_state=0)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    
    if scale:
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        scaler = preprocessing.StandardScaler().fit(X_test)
        X_test = scaler.transform(X_test)
        
    return X_train, X_test, y_train, y_test


def validation_data(scale, encode, na):
    """
    Prepare validation set
    :param scale:
    :param encode:
    :param na:
    :return:
    """
    
    path = "../data/validation.csv"
    df = pd.read_csv(path)
    df.pop("unique_id")
    
    if encode == "label":
        df = label_encode(df)
    if encode == "onehot":
        df = one_hot_encode(df)
        
    df = preprocess_na(df, method=na)
    X = df.to_numpy()
    
    if scale:
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)

    return X


