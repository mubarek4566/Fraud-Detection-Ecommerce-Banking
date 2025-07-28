# Scripts/models.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    average_precision_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_ensemble_model(X_train, y_train, model_type="xgboost"):
    if model_type == "xgboost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model_type. Choose 'xgboost' or 'random_forest'")
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"=== {model_name} Evaluation ===")
    print("F1 Score:", f1)
    print("AUC-PR:", auc_pr)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'{model_name} (AUC-PR={auc_pr:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "f1": f1,
        "auc_pr": auc_pr,
        "confusion_matrix": cm
    }
