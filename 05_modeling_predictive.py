"""
CRISP-DM Phase 4b: Predictive Modeling
Credit Card Routing Optimization Project

This script implements advanced machine learning models for intelligent PSP routing
to optimize both success rates and costs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PredictiveModel:
    """Advanced Predictive Models for PSP Routing Optimization"""
    
    def __init__(self, data_path='input/PSP_Jan_Feb_2019.xlsx'):
        self.data_path = data_path
        self.models = {}
        self.model_performance = {}
        self.feature_importance = {}
        self.best_model = None
        
        # PSP fee structure
        self.psp_fees = {
            'Moneycard': {'success': 5.00, 'failure': 2.00},
            'Goldcard': {'success': 10.00, 'failure': 5.00},
            'UK_Card': {'success': 3.00, 'failure': 1.00},
            'Simplecard': {'success': 1.00, 'failure': 0.50}
        }
        
        # PSP mappings for multi-class classification
        self.psp_to_int = {'Moneycard': 0, 'Goldcard': 1, 'UK_Card': 2, 'Simplecard': 3}
        self.int_to_psp = {v: k for k, v in self.psp_to_int.items()}
    
    def prepare_data(self):
        """Load and prepare data for predictive modeling"""
        print("=== PREPARING DATA FOR PREDICTIVE MODELING ===")
        
        # Load data
        df = pd.read_excel(self.data_path)
        df['tmsp'] = pd.to_datetime(df['tmsp'])
        
        # Feature engineering (simplified version)
        df['hour'] = df['tmsp'].dt.hour
        df['day_of_week'] = df['tmsp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['amount_log'] = np.log1p(df['amount'])
        df['is_high_value'] = (df['amount'] > df['amount'].quantile(0.8)).astype(int)
        df['3D_secured'] = df['3D_secured'].fillna(0).astype(int)
        
        # Create PSP performance features
        psp_success_rates = df.groupby('PSP')['success'].mean()
        df['psp_historical_success'] = df['PSP'].map(psp_success_rates)
        
        # Add PSP costs
        df['psp_success_fee'] = df['PSP'].map(lambda x: self.psp_fees[x]['success'])
        df['psp_failure_fee'] = df['PSP'].map(lambda x: self.psp_fees[x]['failure'])
        
        # Encode categorical variables
        df['country_encoded'] = pd.Categorical(df['country']).codes
        df['card_encoded'] = pd.Categorical(df['card']).codes
        df['psp_encoded'] = df['PSP'].map(self.psp_to_int)
        
        # Select features for modeling
        feature_columns = [
            'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
            'amount', 'amount_log', 'is_high_value',
            '3D_secured', 'country_encoded', 'card_encoded',
            'psp_success_fee', 'psp_failure_fee'
        ]
        
        X = df[feature_columns].fillna(0)
        y_success = df['success']  # For success prediction
        y_psp = df['psp_encoded']  # For PSP recommendation
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_success_train, y_success_test, y_psp_train, y_psp_test = train_test_split(
            X, y_success, y_psp, test_size=0.2, random_state=42, stratify=y_success
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {len(feature_columns)}")
        
        return X_train, X_test, y_success_train, y_success_test, y_psp_train, y_psp_test, feature_columns, df
    
    def train_success_prediction_models(self, X_train, X_test, y_train, y_test, feature_names):
        """Train models to predict transaction success"""
        print("\n=== TRAINING SUCCESS PREDICTION MODELS ===")
        
        # Define models
        models = {
            'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        }
        
        model_results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(X_test, y_test)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                feature_imp = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                feature_imp = None
            
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba,
                'feature_importance': feature_imp
            }
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  AUC Score: {auc_score:.3f}")
            print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Find best model
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc_score'])
        best_model = model_results[best_model_name]
        
        print(f"\n=== BEST SUCCESS PREDICTION MODEL ===")
        print(f"Winner: {best_model_name}")
        print(f"AUC Score: {best_model['auc_score']:.3f}")
        print(f"Accuracy: {best_model['accuracy']:.3f}")
        
        self.success_models = model_results
        self.best_success_model = best_model_name
        
        return model_results, best_model_name
    
    def create_psp_recommendation_system(self, X_train, X_test, feature_names, df):
        """Create intelligent PSP recommendation system"""
        print("\n=== CREATING PSP RECOMMENDATION SYSTEM ===")
        
        # Strategy: For each transaction, predict success probability for each PSP
        # and recommend the PSP with best cost-benefit ratio
        
        psp_models = {}
        
        for psp_name, psp_code in self.psp_to_int.items():
            print(f"Training model for {psp_name}...")
            
            # Create binary target: was this PSP used and was it successful?
            psp_mask = (df['PSP'] == psp_name)
            psp_success = df[psp_mask]['success'] if psp_mask.sum() > 0 else pd.Series([0])
            
            if len(psp_success) > 50:  # Only train if sufficient data
                # Get features for this PSP's transactions
                psp_X = df[psp_mask][feature_names].fillna(0)
                
                if len(psp_X) > 0:
                    # Train success prediction model for this PSP
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(psp_X, psp_success)
                    
                    psp_models[psp_name] = {
                        'model': model,
                        'success_rate': psp_success.mean(),
                        'transaction_count': len(psp_success)
                    }
        
        self.psp_models = psp_models
        
        print(f"Trained PSP-specific models for {len(psp_models)} PSPs")
        return psp_models
    
    def recommend_optimal_psp(self, transaction_features):
        """Recommend optimal PSP for a given transaction"""
        
        recommendations = {}
        
        for psp_name, psp_info in self.psp_models.items():
            # Predict success probability
            success_prob = psp_info['model'].predict_proba(transaction_features.reshape(1, -1))[0][1]
            
            # Calculate expected cost
            expected_cost = (success_prob * self.psp_fees[psp_name]['success'] + 
                           (1 - success_prob) * self.psp_fees[psp_name]['failure'])
            
            # Calculate expected revenue (assuming 2.5% commission on successful transactions)
            commission_rate = 0.025
            transaction_amount = transaction_features[4]  # amount is 5th feature
            expected_revenue = success_prob * transaction_amount * commission_rate
            
            # Net expected value
            net_value = expected_revenue - expected_cost
            
            recommendations[psp_name] = {
                'success_probability': success_prob,
                'expected_cost': expected_cost,
                'expected_revenue': expected_revenue,
                'net_expected_value': net_value
            }
        
        # Recommend PSP with highest net expected value
        best_psp = max(recommendations.keys(), key=lambda x: recommendations[x]['net_expected_value'])
        
        return best_psp, recommendations
    
    def evaluate_psp_recommendation_system(self, X_test, df):
        """Evaluate the PSP recommendation system"""
        print("\n=== EVALUATING PSP RECOMMENDATION SYSTEM ===")
        
        # Get test data with original PSP information
        test_indices = X_test.index
        test_df = df.loc[test_indices].copy()
        
        recommendations = []
        
        for idx, row in X_test.iterrows():
            try:
                best_psp, psp_recommendations = self.recommend_optimal_psp(row.values)
                recommendations.append({
                    'transaction_id': idx,
                    'recommended_psp': best_psp,
                    'actual_psp': df.loc[idx, 'PSP'],
                    'actual_success': df.loc[idx, 'success'],
                    'recommendations': psp_recommendations
                })
            except Exception as e:
                # Handle cases where prediction fails
                recommendations.append({
                    'transaction_id': idx,
                    'recommended_psp': 'Simplecard',  # Default to cheapest
                    'actual_psp': df.loc[idx, 'PSP'],
                    'actual_success': df.loc[idx, 'success'],
                    'recommendations': {}
                })
        
        recommendation_df = pd.DataFrame(recommendations)
        
        # Calculate improvement metrics
        # Simulate performance of recommended PSPs
        simulated_improvements = []
        
        for _, rec in recommendation_df.iterrows():
            if rec['recommendations']:
                recommended_psp = rec['recommended_psp']
                recommended_success_prob = rec['recommendations'][recommended_psp]['success_probability']
                
                # Use historical success rate as proxy for actual performance
                historical_rate = self.psp_models[recommended_psp]['success_rate']
                
                simulated_improvements.append({
                    'recommended_success_prob': recommended_success_prob,
                    'historical_success_rate': historical_rate,
                    'actual_success': rec['actual_success'],
                    'recommended_psp': recommended_psp,
                    'actual_psp': rec['actual_psp']
                })
        
        improvement_df = pd.DataFrame(simulated_improvements)
        
        # Calculate aggregate improvements
        avg_recommended_success = improvement_df['historical_success_rate'].mean()
        avg_actual_success = improvement_df['actual_success'].mean()
        success_rate_improvement = avg_recommended_success - avg_actual_success
        
        print(f"Average recommended PSP success rate: {avg_recommended_success:.1%}")
        print(f"Average actual success rate: {avg_actual_success:.1%}")
        print(f"Potential success rate improvement: {success_rate_improvement:+.1%}")
        
        return recommendation_df, improvement_df
    
    def analyze_feature_importance(self):
        """Analyze feature importance across models"""
        print("\n=== ANALYZING FEATURE IMPORTANCE ===")
        
        # Get feature importance from best success prediction model
        best_model_results = self.success_models[self.best_success_model]
        
        if best_model_results['feature_importance'] is not None:
            feature_imp = best_model_results['feature_importance']
            
            # Create feature importance visualization
            plt.figure(figsize=(10, 8))
            top_features = feature_imp.head(10)
            
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 10 Features - {self.best_success_model}', fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig('results/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Top 5 Most Important Features:")
            for i, row in feature_imp.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return feature_imp if best_model_results['feature_importance'] is not None else None
    
    def create_model_comparison_visualization(self):
        """Create comprehensive model comparison visualization"""
        print("\n=== CREATING MODEL COMPARISON VISUALIZATION ===")
        
        # Prepare data for visualization
        model_names = list(self.success_models.keys())
        accuracies = [self.success_models[name]['accuracy'] for name in model_names]
        auc_scores = [self.success_models[name]['auc_score'] for name in model_names]
        cv_scores = [self.success_models[name]['cv_mean'] for name in model_names]
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Accuracy Comparison
        colors = ['gold' if name == self.best_success_model else 'lightblue' for name in model_names]
        bars1 = ax1.bar(model_names, accuracies, color=colors)
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. AUC Score Comparison
        bars2 = ax2.bar(model_names, auc_scores, color=colors)
        ax2.set_title('AUC Score Comparison', fontweight='bold')
        ax2.set_ylabel('AUC Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Cross-Validation Scores
        bars3 = ax3.bar(model_names, cv_scores, color=colors)
        ax3.set_title('Cross-Validation Score Comparison', fontweight='bold')
        ax3.set_ylabel('CV Score')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. ROC Curve for Best Model
        best_model = self.success_models[self.best_success_model]
        
        # Calculate ROC curve (using stored test data)
        if hasattr(self, 'y_test') and hasattr(self, 'X_test'):
            y_pred_proba = best_model['model'].predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            
            ax4.plot(fpr, tpr, linewidth=2, label=f'{self.best_success_model} (AUC = {best_model["auc_score"]:.3f})')
            ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax4.set_xlabel('False Positive Rate')
            ax4.set_ylabel('True Positive Rate')
            ax4.set_title('ROC Curve - Best Model', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/predictive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_predictive_modeling(self):
        """Execute complete predictive modeling phase"""
        print("Starting CRISP-DM Phase 4b: Predictive Modeling")
        print("=" * 50)
        
        # Prepare data
        X_train, X_test, y_success_train, y_success_test, y_psp_train, y_psp_test, feature_names, df = self.prepare_data()
        
        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_success_test
        
        # Train success prediction models
        success_models, best_success_model = self.train_success_prediction_models(
            X_train, X_test, y_success_train, y_success_test, feature_names
        )
        
        # Create PSP recommendation system
        psp_models = self.create_psp_recommendation_system(X_train, X_test, feature_names, df)
        
        # Evaluate recommendation system
        recommendation_results, improvement_results = self.evaluate_psp_recommendation_system(X_test, df)
        
        # Analyze feature importance
        feature_importance = self.analyze_feature_importance()
        
        # Create visualizations
        self.create_model_comparison_visualization()
        
        # Compile results
        predictive_results = {
            'success_models': success_models,
            'best_success_model': best_success_model,
            'psp_models': psp_models,
            'recommendation_results': recommendation_results,
            'improvement_results': improvement_results,
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'modeling_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n=== PREDICTIVE MODELING COMPLETE ===")
        print("Key achievements:")
        print(f"• {len(success_models)} success prediction models trained")
        print(f"• Best model: {best_success_model} (AUC: {success_models[best_success_model]['auc_score']:.3f})")
        print(f"• {len(psp_models)} PSP-specific models created")
        print(f"• Intelligent recommendation system implemented")
        
        return predictive_results

def main():
    """Main execution function"""
    predictive = PredictiveModel()
    results = predictive.run_predictive_modeling()
    return results

if __name__ == "__main__":
    results = main() 