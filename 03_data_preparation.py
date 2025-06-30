"""
CRISP-DM Phase 3: Data Preparation
Credit Card Routing Optimization Project

This script performs feature engineering, data preprocessing, and preparation
for machine learning models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreparation:
    """Data Preparation Phase - Feature engineering and preprocessing"""
    
    def __init__(self, data_path='input/PSP_Jan_Feb_2019.xlsx'):
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # PSP fee structure for cost calculations
        self.psp_fees = {
            'Moneycard': {'success': 5.00, 'failure': 2.00},
            'Goldcard': {'success': 10.00, 'failure': 5.00},
            'UK_Card': {'success': 3.00, 'failure': 1.00},
            'Simplecard': {'success': 1.00, 'failure': 0.50}
        }
    
    def load_and_clean_data(self):
        """Load data and perform basic cleaning"""
        print("=== LOADING AND CLEANING DATA ===")
        
        self.df = pd.read_excel(self.data_path)
        print(f"Original data shape: {self.df.shape}")
        
        # Remove any duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Removed {initial_count - len(self.df)} duplicate records")
        
        # Remove rows with missing critical values
        critical_columns = ['tmsp', 'country', 'amount', 'success', 'PSP']
        missing_critical = self.df[critical_columns].isnull().any(axis=1)
        self.df = self.df[~missing_critical]
        print(f"Removed {missing_critical.sum()} rows with missing critical data")
        
        # Basic data type corrections
        self.df['tmsp'] = pd.to_datetime(self.df['tmsp'])
        self.df['amount'] = pd.to_numeric(self.df['amount'], errors='coerce')
        self.df['success'] = self.df['success'].astype(int)
        self.df['3D_secured'] = self.df['3D_secured'].fillna(0).astype(int)
        
        print(f"Clean data shape: {self.df.shape}")
        return self.df
    
    def create_temporal_features(self):
        """Create time-based features"""
        print("\n=== CREATING TEMPORAL FEATURES ===")
        
        # Extract datetime components
        self.df['hour'] = self.df['tmsp'].dt.hour
        self.df['day_of_week'] = self.df['tmsp'].dt.dayofweek
        self.df['day_of_month'] = self.df['tmsp'].dt.day
        self.df['month'] = self.df['tmsp'].dt.month
        
        # Business hours indicator
        self.df['is_business_hours'] = ((self.df['hour'] >= 9) & (self.df['hour'] <= 17)).astype(int)
        
        # Weekend indicator
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # Peak hours (based on typical e-commerce patterns)
        self.df['is_peak_hours'] = ((self.df['hour'] >= 19) & (self.df['hour'] <= 22)).astype(int)
        
        print("Created temporal features: hour, day_of_week, day_of_month, month, is_business_hours, is_weekend, is_peak_hours")
        
    def create_amount_features(self):
        """Create amount-based features"""
        print("\n=== CREATING AMOUNT FEATURES ===")
        
        # Amount categories
        self.df['amount_log'] = np.log1p(self.df['amount'])
        
        # Amount quartiles
        amount_quartiles = self.df['amount'].quantile([0.25, 0.5, 0.75])
        self.df['amount_quartile'] = pd.cut(self.df['amount'], 
                                           bins=[-np.inf, amount_quartiles[0.25], 
                                                amount_quartiles[0.5], amount_quartiles[0.75], np.inf],
                                           labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # High value transaction indicator
        high_value_threshold = self.df['amount'].quantile(0.9)
        self.df['is_high_value'] = (self.df['amount'] > high_value_threshold).astype(int)
        
        # Micro transaction indicator
        micro_threshold = self.df['amount'].quantile(0.1)
        self.df['is_micro_transaction'] = (self.df['amount'] < micro_threshold).astype(int)
        
        print(f"Created amount features: amount_log, amount_quartile, is_high_value (>{high_value_threshold:.2f}), is_micro_transaction (<{micro_threshold:.2f})")
    
    def create_psp_features(self):
        """Create PSP-related features"""
        print("\n=== CREATING PSP FEATURES ===")
        
        # PSP success fees
        self.df['psp_success_fee'] = self.df['PSP'].map(lambda x: self.psp_fees[x]['success'])
        self.df['psp_failure_fee'] = self.df['PSP'].map(lambda x: self.psp_fees[x]['failure'])
        
        # Fee difference (cost of failure vs success)
        self.df['psp_fee_difference'] = self.df['psp_success_fee'] - self.df['psp_failure_fee']
        
        # Expected cost based on current success rate
        psp_success_rates = self.df.groupby('PSP')['success'].mean()
        self.df['psp_historical_success_rate'] = self.df['PSP'].map(psp_success_rates)
        
        # Expected cost calculation
        self.df['expected_cost'] = (self.df['psp_historical_success_rate'] * self.df['psp_success_fee'] + 
                                   (1 - self.df['psp_historical_success_rate']) * self.df['psp_failure_fee'])
        
        # PSP cost tier (cheap, medium, expensive)
        try:
            cost_bins = pd.qcut(self.df['expected_cost'], q=3, labels=['cheap', 'medium', 'expensive'], duplicates='drop')
            self.df['psp_cost_tier'] = cost_bins
        except ValueError:
            # If still issues, use cut instead of qcut
            cost_bins = pd.cut(self.df['expected_cost'], bins=3, labels=['cheap', 'medium', 'expensive'])
            self.df['psp_cost_tier'] = cost_bins
        
        print("Created PSP features: psp_success_fee, psp_failure_fee, psp_fee_difference, psp_historical_success_rate, expected_cost, psp_cost_tier")
    
    def identify_retry_attempts(self):
        """Identify and create features for retry attempts"""
        print("\n=== IDENTIFYING RETRY ATTEMPTS ===")
        
        # Sort by country, amount, and timestamp for retry detection
        df_sorted = self.df.sort_values(['country', 'amount', 'tmsp'])
        
        # Calculate time differences between transactions with same country/amount
        df_sorted['time_diff'] = df_sorted.groupby(['country', 'amount'])['tmsp'].diff()
        
        # Mark potential retries (within 5 minutes of previous transaction)
        df_sorted['is_potential_retry'] = ((df_sorted['time_diff'] <= timedelta(minutes=5)) & 
                                          (df_sorted['time_diff'].notna())).astype(int)
        
        # Count of attempts for same country/amount combination
        df_sorted['attempt_number'] = df_sorted.groupby(['country', 'amount']).cumcount() + 1
        
        # Time since last attempt (in minutes)
        df_sorted['minutes_since_last_attempt'] = df_sorted['time_diff'].dt.total_seconds() / 60
        df_sorted['minutes_since_last_attempt'] = df_sorted['minutes_since_last_attempt'].fillna(0)
        
        # Merge back to original dataframe
        self.df = df_sorted.sort_index()
        
        retry_count = self.df['is_potential_retry'].sum()
        print(f"Identified {retry_count} potential retry attempts ({retry_count/len(self.df)*100:.1f}% of transactions)")
    
    def create_interaction_features(self):
        """Create interaction features between important variables"""
        print("\n=== CREATING INTERACTION FEATURES ===")
        
        # Country-PSP combinations (some PSPs might work better in certain countries)
        self.df['country_psp_combo'] = self.df['country'] + '_' + self.df['PSP']
        
        # Card-PSP combinations
        self.df['card_psp_combo'] = self.df['card'].astype(str) + '_' + self.df['PSP']
        
        # 3D Secured interaction with high-value transactions
        self.df['secured_high_value'] = self.df['3D_secured'] * self.df['is_high_value']
        
        # Amount category and PSP interaction
        self.df['amount_quartile_psp'] = self.df['amount_quartile'].astype(str) + '_' + self.df['PSP']
        
        print("Created interaction features: country_psp_combo, card_psp_combo, secured_high_value, amount_quartile_psp")
    
    def encode_categorical_features(self):
        """Encode categorical features for machine learning"""
        print("\n=== ENCODING CATEGORICAL FEATURES ===")
        
        categorical_features = ['country', 'PSP', 'card', 'amount_quartile', 'psp_cost_tier',
                               'country_psp_combo', 'card_psp_combo', 'amount_quartile_psp']
        
        for feature in categorical_features:
            if feature in self.df.columns:
                le = LabelEncoder()
                # Handle any NaN values
                self.df[feature] = self.df[feature].fillna('Unknown')
                self.df[f'{feature}_encoded'] = le.fit_transform(self.df[feature])
                self.label_encoders[feature] = le
        
        print(f"Encoded {len(categorical_features)} categorical features")
    
    def create_target_features(self):
        """Create additional target-related features"""
        print("\n=== CREATING TARGET FEATURES ===")
        
        # Calculate actual transaction cost based on success/failure
        self.df['actual_cost'] = np.where(self.df['success'] == 1,
                                         self.df['psp_success_fee'],
                                         self.df['psp_failure_fee'])
        
        # Revenue impact (for successful transactions, assume commission)
        avg_commission_rate = 0.025  # 2.5% commission assumption
        self.df['revenue_impact'] = np.where(self.df['success'] == 1,
                                           self.df['amount'] * avg_commission_rate,
                                           -self.df['psp_failure_fee'])  # Lost opportunity cost
        
        # Net profit per transaction
        self.df['net_profit'] = self.df['revenue_impact'] - self.df['actual_cost']
        
        print("Created target features: actual_cost, revenue_impact, net_profit")
    
    def prepare_model_datasets(self):
        """Prepare final datasets for modeling"""
        print("\n=== PREPARING MODEL DATASETS ===")
        
        # Select features for modeling
        feature_columns = [
            # Temporal features
            'hour', 'day_of_week', 'day_of_month', 'month',
            'is_business_hours', 'is_weekend', 'is_peak_hours',
            
            # Amount features
            'amount', 'amount_log', 'is_high_value', 'is_micro_transaction',
            
            # PSP features
            'psp_success_fee', 'psp_failure_fee', 'psp_fee_difference',
            'psp_historical_success_rate', 'expected_cost',
            
            # Other features
            '3D_secured', 'is_potential_retry', 'attempt_number',
            'minutes_since_last_attempt', 'secured_high_value',
            
            # Encoded categorical features
            'country_encoded', 'PSP_encoded', 'card_encoded',
            'amount_quartile_encoded', 'psp_cost_tier_encoded'
        ]
        
        # Filter existing columns
        available_features = [col for col in feature_columns if col in self.df.columns]
        
        # Create feature matrix
        X = self.df[available_features].copy()
        y = self.df['success'].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_features = X.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Feature distribution - Success: {y.mean():.1%}, Failure: {1-y.mean():.1%}")
        
        # Store processed datasets
        datasets = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'feature_names': available_features,
            'processed_df': self.df
        }
        
        return datasets
    
    def generate_feature_summary(self):
        """Generate summary of created features"""
        print("\n=== FEATURE ENGINEERING SUMMARY ===")
        
        feature_categories = {
            'Temporal': ['hour', 'day_of_week', 'day_of_month', 'month', 
                        'is_business_hours', 'is_weekend', 'is_peak_hours'],
            'Amount-based': ['amount_log', 'amount_quartile', 'is_high_value', 'is_micro_transaction'],
            'PSP-related': ['psp_success_fee', 'psp_failure_fee', 'psp_fee_difference',
                           'psp_historical_success_rate', 'expected_cost', 'psp_cost_tier'],
            'Retry Detection': ['is_potential_retry', 'attempt_number', 'minutes_since_last_attempt'],
            'Interactions': ['country_psp_combo', 'card_psp_combo', 'secured_high_value', 'amount_quartile_psp'],
            'Business Impact': ['actual_cost', 'revenue_impact', 'net_profit']
        }
        
        total_features = 0
        for category, features in feature_categories.items():
            existing_features = [f for f in features if f in self.df.columns]
            print(f"{category}: {len(existing_features)} features")
            total_features += len(existing_features)
        
        print(f"\nTotal engineered features: {total_features}")
        print(f"Original columns: {len(pd.read_excel(self.data_path).columns)}")
        print(f"Final columns: {len(self.df.columns)}")
        
        return feature_categories
    
    def run_data_preparation(self):
        """Execute complete data preparation phase"""
        print("Starting CRISP-DM Phase 3: Data Preparation")
        print("=" * 50)
        
        # Execute all data preparation steps
        self.load_and_clean_data()
        self.create_temporal_features()
        self.create_amount_features()
        self.create_psp_features()
        self.identify_retry_attempts()
        self.create_interaction_features()
        self.encode_categorical_features()
        self.create_target_features()
        
        # Prepare final datasets
        datasets = self.prepare_model_datasets()
        feature_summary = self.generate_feature_summary()
        
        # Compile preparation results
        preparation_results = {
            'datasets': datasets,
            'feature_summary': feature_summary,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'processed_df': self.df,
            'preparation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n=== DATA PREPARATION COMPLETE ===")
        print("Data ready for modeling with comprehensive feature engineering")
        print(f"Final feature count: {len(datasets['feature_names'])}")
        
        return preparation_results

def main():
    """Main execution function"""
    data_prep = DataPreparation()
    results = data_prep.run_data_preparation()
    return results

if __name__ == "__main__":
    results = main() 