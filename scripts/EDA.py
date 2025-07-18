import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")


def perform_univariate_analysis(df):
    """Univariate analysis on key columns of fraud dataset."""
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))

    # Age distribution
    sns.histplot(df["age"], kde=True, bins=20, ax=axs[0, 0])
    axs[0, 0].set_title("Age Distribution")

    # Purchase value distribution
    sns.histplot(df["purchase_value"], kde=True, bins=20, ax=axs[0, 1])
    axs[0, 1].set_title("Purchase Value Distribution")

    # Class distribution
    sns.countplot(x="class", data=df, ax=axs[1, 0])
    axs[1, 0].set_title("Fraud vs Non-Fraud Count")
    axs[1, 0].set_xticklabels(['Non-Fraud (0)', 'Fraud (1)'])

    # Browser usage
    sns.countplot(y="browser", data=df, order=df["browser"].value_counts().index, ax=axs[1, 1])
    axs[1, 1].set_title("Browser Usage")

    # Source breakdown
    sns.countplot(x="source", data=df, ax=axs[2, 0])
    axs[2, 0].set_title("Traffic Source")

    # Sex distribution
    sns.countplot(x="sex", data=df, ax=axs[2, 1])
    axs[2, 1].set_title("Gender Distribution")

    plt.tight_layout()
    plt.show()


def perform_bivariate_analysis(df):
    """Bivariate analysis comparing features with fraud class."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Age vs Class
    sns.boxplot(x="class", y="age", data=df, ax=axs[0, 0])
    axs[0, 0].set_title("Age vs Fraud Class")

    # Purchase Value vs Class
    sns.boxplot(x="class", y="purchase_value", data=df, ax=axs[0, 1])
    axs[0, 1].set_title("Purchase Value vs Fraud Class")

    # Source vs Class
    sns.countplot(x="source", hue="class", data=df, ax=axs[1, 0])
    axs[1, 0].set_title("Traffic Source vs Fraud Class")

    # Browser vs Class
    sns.countplot(y="browser", hue="class", data=df, ax=axs[1, 1])
    axs[1, 1].set_title("Browser vs Fraud Class")

    plt.tight_layout()
    plt.show()

def univariate_analysis(df, column):
    """Perform univariate analysis for any column."""
    plt.figure(figsize=(10, 6))
    if pd.api.types.is_numeric_dtype(df[column]):
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f"Distribution of {column}")
    else:
        sns.countplot(x=column, data=df, order=df[column].value_counts().index)
        plt.title(f"Count Plot of {column}")
        plt.xticks(rotation=45)
    plt.xlabel(column)
    plt.tight_layout()
    plt.show()


def bivariate_analysis(df, feature, target):
    """Perform bivariate analysis between feature and target."""
    plt.figure(figsize=(10, 6))
    if pd.api.types.is_numeric_dtype(df[feature]):
        sns.boxplot(x=target, y=feature, data=df)
        plt.title(f"{feature} vs {target} (Box Plot)")
    else:
        sns.countplot(x=feature, hue=target, data=df, order=df[feature].value_counts().index)
        plt.title(f"{feature} vs {target} (Count Plot)")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
