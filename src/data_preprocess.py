import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


def read_data(path):
    df = pd.read_csv(path)
    y = df.pop('nEvent')
    X = df
    return X, y


def encode(df):
    label_encoder = preprocessing.LabelEncoder()
    df['diet'] = label_encoder.fit_transform(df['diet'])
    for num in "1234":
        df[f'MachineryVar{num}'] = label_encoder.fit_transform(df[f'MachineryVar{num}'])
    return df


def preprocess_na(df, method):
    if method == "zero":
        df = df.fillna(0)
    if method == "mean":
        df = df.fillna(df.mean())
    return df


def training_data(scale):
    path = "../data/data.csv"
    X, y = read_data(path)
    X = encode(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = preprocess_na(X_train, method="zero").to_numpy()
    X_test = preprocess_na(X_test, method="zero").to_numpy()
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


def validation_data():
    path = "../data/data.csv"
    X = pd.read_csv(path)
    X = preprocess_na(encode(X), method="zero").to_numpy()
    return X


