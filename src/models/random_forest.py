from src.data_preprocess import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


X_train, X_test, y_train, y_test = training_data(scale=True)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(y_train)

train_score = roc_auc_score(y_train, y_train_pred, multi_class='ovr')
test_score = roc_auc_score(y_test, y_test_pred, multi_class='ovr')

if __name__ == "__main__":
    print(train_score)
    print(test_score)