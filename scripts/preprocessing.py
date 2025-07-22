import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek


class preprocess:
    def __init__(self, path):
        self.df = path 

    def engineer_features(self):
        """
        Perform feature engineering on the fraud dataset.
        """
        # Make a copy to avoid modifying the original dataframe
        self.df = self.df.copy()
        
        # 1. Transaction Frequency and Velocity features
        print("Engineering transaction frequency and velocity features...")
        
        # Sort by user_id and purchase_time
        self.df = self.df.sort_values(['user_id', 'purchase_time'])
        
        # Calculate time since last transaction (velocity)
        self.df['time_since_last_txn'] = self.df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 60  # in minutes
        
        # Calculate transaction count (frequency)
        self.df['txn_count'] = self.df.groupby('user_id').cumcount() + 1
        
        # 2. Time-Based Features
        print("Engineering time-based features...")
        
        # Extract hour of day from purchase time
        self.df['hour_of_day'] = self.df['purchase_time'].dt.hour
        
        # Extract day of week (Monday=0, Sunday=6)
        self.df['day_of_week'] = self.df['purchase_time'].dt.dayofweek
        
        # Create time of day categories
        bins = [0, 6, 12, 18, 24]
        labels = ['Night', 'Morning', 'Afternoon', 'Evening']
        self.df['time_of_day'] = pd.cut(self.df['hour_of_day'], bins=bins, labels=labels, right=False)
        
        # 3. Time Since Signup
        print("Calculating time since signup...")
        self.df['time_since_signup'] = (self.df['purchase_time'] - self.df['signup_time']).dt.total_seconds() / 3600  # in hours
        
        # Additional useful features
        print("Creating additional features...")
        
        # Weekend flag
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        
        # Browser popularity feature
        browser_counts = self.df['browser'].value_counts(normalize=True)
        self.df['browser_popularity'] = self.df['browser'].map(browser_counts)
        
        # Convert new categorical features
        self.df['time_of_day'] = self.df['time_of_day'].astype('category')
        
        return self.df

    def split_data(self, target_col='class', test_size=0.3, random_state=42):
        """
        Split data into train and test sets while preserving class distribution.
        """
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def apply_smote(X_train, y_train, random_state=42):
        """
        Apply SMOTE oversampling to the training data.
        """
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print("\nAfter SMOTE:")
        print(pd.Series(y_resampled).value_counts())
        
        return X_resampled, y_resampled


    def apply_undersampling(X_train, y_train, random_state=42):
        """
        Apply random undersampling to the training data.
        """
        rus = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        
        print("\nAfter Undersampling:")
        print(pd.Series(y_resampled).value_counts())
        
        return X_resampled, y_resampled

    def apply_smotetomek(X_train, y_train, random_state=42):
        """
        Apply SMOTE + Tomek Links combination to the training data.
        """
        smote_tomek = SMOTETomek(random_state=random_state)
        X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
        
        print("\nAfter SMOTE + Tomek Links:")
        print(pd.Series(y_resampled).value_counts())
        
        return X_resampled, y_resampled
    
    