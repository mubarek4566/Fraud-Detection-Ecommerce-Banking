import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(style="whitegrid")

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

def merge_datasets(fraud_df, ip_df):
    """
    Merge fraud dataset with IP address country mapping dataset.
    """
    
    # Convert IP addresses to numeric format in both datasets
    fraud_df['ip_numeric'] = fraud_df['ip_address'].astype(float)
    
    # Sort the IP address ranges for efficient searching
    ip_df = ip_df.sort_values('lower_bound_ip_address')
    
    # Function to find country for a given IP
    def find_country(ip_num):
        # Find the first range where lower_bound <= ip_num <= upper_bound
        mask = (ip_df['lower_bound_ip_address'] <= ip_num) & (ip_num <= ip_df['upper_bound_ip_address'])
        matches = ip_df[mask]
        if len(matches) > 0:
            return matches.iloc[0]['country']
        return None
    
    # Apply the function to find countries for each IP
    fraud_df['country'] = fraud_df['ip_numeric'].apply(find_country)
    
    # Convert country to categorical to save memory
    fraud_df['country'] = fraud_df['country'].astype('category')
    
    return fraud_df

