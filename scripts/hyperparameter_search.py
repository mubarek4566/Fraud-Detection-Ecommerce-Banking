# scripts/hyperparameter_search.py

import json
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def search_best_params_rf(X_train, y_train, random_state=42):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=random_state)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    with open("best_rf_params.json", "w") as f:
        json.dump(best_params, f)
    return best_params

def search_best_params_xgb(X_train, y_train, random_state=42):
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, cv=3, scoring='f1', n_jobs=-1, random_state=random_state)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_

    with open("best_xgb_params.json", "w") as f:
        json.dump(best_params, f)
    return best_params
