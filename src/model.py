from data_preprocess import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import *
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


def train_and_predict(model, encode, na):

    if model == "rf":
        scale = False
        clf = RandomForestClassifier(n_estimators=260, criterion="entropy", min_samples_leaf=16, max_depth=26, random_state=0)
    if model == "nb":
        scale = False
        clf = GaussianNB()
    if model == "mlp":
        scale = False
        clf = MLPClassifier(random_state=1, max_iter=30000, momentum=0.99, batch_size=1000, solver="adam")
    if model == "lr":
        scale = True
        clf = LogisticRegression(random_state=0, max_iter=10000)

    X_train, X_test, y_train, y_test = training_data(scale=scale, encode=encode, na=na)
    validation_set = validation_data(scale=scale, encode=encode, na=na)

    clf.fit(X_train, y_train)
    y_train_pred = clf.predict_proba(X_train)
    y_train_pred = y_train_pred[:, 1]
    y_test_pred = clf.predict_proba(X_test)
    y_test_pred = y_test_pred[:, 1]

    train_score = roc_auc_score(y_train, y_train_pred, multi_class='ovr')
    test_score = roc_auc_score(y_test, y_test_pred, multi_class='ovr')

    result = clf.predict_proba(validation_set)
    result = result[:, 1]
    
    return train_score, test_score, result
