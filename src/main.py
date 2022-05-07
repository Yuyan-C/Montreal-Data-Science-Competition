from data_preprocess import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = training_data(scale=True)
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf.fit(X_train, y_train)
# y_train_pred = clf.predict(X_train)
# y_test_pred = clf.predict(X_test)

# clf = MLPClassifier(random_state=1, max_iter=30000).fit(X_train, y_train)
clf = LogisticRegression(random_state=0, max_iter=10000).fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_train_pred = gnb.predict(X_train)
# y_test_pred = gnb.predict(X_test)


train_score = roc_auc_score(y_train, y_train_pred, multi_class='ovr')
test_score = roc_auc_score(y_test, y_test_pred, multi_class='ovr')

if __name__ == "__main__":
    print(train_score)
    print(test_score)
    print(classification_report(y_train, y_train_pred))
    print(classification_report(y_test, y_test_pred))