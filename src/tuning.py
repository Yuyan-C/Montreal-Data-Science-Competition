from model import *

from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap': [True],
    'max_depth': [20, 25, 30],
    'max_features': [2, 3],
    'min_samples_leaf': [10, 15, 20, 25],
    'min_samples_split': [2, 3, 4, 5, 6],
    'n_estimators': [1150, 1200, 1250, 1300, 1350]
}

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid)

X_train, X_test, y_train, y_test = training_data(scale=False, encode="label", na="zero")
grid_search.fit(X_train, y_train)

if __name__ == "__main__":
    print(grid_search.best_params_)

