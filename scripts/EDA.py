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


def engineer_features(df):
    """
    Perform feature engineering on the fraud dataset.
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # 1. Transaction Frequency and Velocity features
    print("Engineering transaction frequency and velocity features...")
    
    # Sort by user_id and purchase_time
    df = df.sort_values(['user_id', 'purchase_time'])
    
    # Calculate time since last transaction (velocity)
    df['time_since_last_txn'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 60  # in minutes
    
    # Calculate transaction count (frequency)
    df['txn_count'] = df.groupby('user_id').cumcount() + 1
    
    # 2. Time-Based Features
    print("Engineering time-based features...")
    
    # Extract hour of day from purchase time
    df['hour_of_day'] = df['purchase_time'].dt.hour
    
    # Extract day of week (Monday=0, Sunday=6)
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Create time of day categories
    bins = [0, 6, 12, 18, 24]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    df['time_of_day'] = pd.cut(df['hour_of_day'], bins=bins, labels=labels, right=False)
    
    # 3. Time Since Signup
    print("Calculating time since signup...")
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600  # in hours
    
    # Additional useful features
    print("Creating additional features...")
    
    # Weekend flag
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Browser popularity feature
    browser_counts = df['browser'].value_counts(normalize=True)
    df['browser_popularity'] = df['browser'].map(browser_counts)
    
    # Convert new categorical features
    df['time_of_day'] = df['time_of_day'].astype('category')
    
    return df
