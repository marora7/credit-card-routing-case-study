"""
CRISP-DM Phase 4a: Baseline Modeling
Credit Card Routing Optimization Project

This script implements a simple rule-based baseline model for PSP selection
to establish performance benchmarks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BaselineModel:
    """Baseline Model - Simple rule-based PSP selection"""
    
    def __init__(self, data_path='input/PSP_Jan_Feb_2019.xlsx'):
        self.data_path = data_path
        self.df = None
        self.baseline_rules = {}
        self.performance_metrics = {}
        
        # PSP fee structure
        self.psp_fees = {
            'Moneycard': {'success': 5.00, 'failure': 2.00},
            'Goldcard': {'success': 10.00, 'failure': 5.00},
            'UK_Card': {'success': 3.00, 'failure': 1.00},
            'Simplecard': {'success': 1.00, 'failure': 0.50}
        }
    
    def load_data(self):
        """Load and prepare data"""
        print("=== LOADING DATA FOR BASELINE MODEL ===")
        
        self.df = pd.read_excel(self.data_path)
        self.df['tmsp'] = pd.to_datetime(self.df['tmsp'])
        
        # Create additional features needed for baseline rules
        self.df['hour'] = self.df['tmsp'].dt.hour
        self.df['is_high_value'] = (self.df['amount'] > self.df['amount'].quantile(0.8)).astype(int)
        
        print(f"Data loaded: {self.df.shape[0]} transactions")
        return True
    
    def analyze_historical_performance(self):
        """Analyze historical PSP performance to inform baseline rules"""
        print("\n=== ANALYZING HISTORICAL PSP PERFORMANCE ===")
        
        # Overall PSP performance
        psp_stats = self.df.groupby('PSP').agg({
            'success': ['count', 'mean'],
            'amount': 'mean'
        }).round(3)
        psp_stats.columns = ['Transaction_Count', 'Success_Rate', 'Avg_Amount']
        
        # Add cost information
        psp_stats['Success_Fee'] = [self.psp_fees[psp]['success'] for psp in psp_stats.index]
        psp_stats['Failure_Fee'] = [self.psp_fees[psp]['failure'] for psp in psp_stats.index]
        
        # Calculate expected cost per transaction
        psp_stats['Expected_Cost'] = (psp_stats['Success_Rate'] * psp_stats['Success_Fee'] + 
                                     (1 - psp_stats['Success_Rate']) * psp_stats['Failure_Fee'])
        
        # Performance by country
        country_psp_performance = self.df.groupby(['country', 'PSP'])['success'].mean().unstack(fill_value=0)
        
        # Performance by transaction value
        self.df['amount_category'] = pd.cut(self.df['amount'], 
                                           bins=[0, 50, 200, 500, np.inf],
                                           labels=['Low', 'Medium', 'High', 'Very_High'])
        
        amount_psp_performance = self.df.groupby(['amount_category', 'PSP'])['success'].mean().unstack(fill_value=0)
        
        # Performance by card type
        card_psp_performance = self.df.groupby(['card', 'PSP'])['success'].mean().unstack(fill_value=0)
        
        print("PSP Performance Summary:")
        print(psp_stats)
        
        # Verify all PSPs are included
        print(f"\nPSPs analyzed: {sorted(psp_stats.index.tolist())}")
        print(f"PSPs in fee structure: {sorted(self.psp_fees.keys())}")
        missing_psps = set(self.psp_fees.keys()) - set(psp_stats.index)
        if missing_psps:
            print(f"WARNING: PSPs in fee structure but not in data: {missing_psps}")
        extra_psps = set(psp_stats.index) - set(self.psp_fees.keys())
        if extra_psps:
            print(f"WARNING: PSPs in data but not in fee structure: {extra_psps}")
        
        # Store performance data for rule creation
        self.historical_performance = {
            'overall': psp_stats,
            'by_country': country_psp_performance,
            'by_amount': amount_psp_performance,
            'by_card': card_psp_performance
        }
        
        return self.historical_performance
    
    def create_baseline_rules(self):
        """Create simple rule-based PSP selection logic"""
        print("\n=== CREATING BASELINE RULES ===")
        
        # Rule 1: High success rate PSP preference
        best_success_psp = self.historical_performance['overall']['Success_Rate'].idxmax()
        
        # Rule 2: Low cost PSP for small transactions
        cheapest_psp = self.historical_performance['overall']['Expected_Cost'].idxmin()
        
        # Rule 3: Country-specific best performers
        country_best_psp = {}
        for country in self.historical_performance['by_country'].index:
            country_performance = self.historical_performance['by_country'].loc[country]
            country_best_psp[country] = country_performance.idxmax()
        
        # Rule 4: Amount-based selection
        amount_best_psp = {}
        for amount_cat in self.historical_performance['by_amount'].index:
            amount_performance = self.historical_performance['by_amount'].loc[amount_cat]
            amount_best_psp[amount_cat] = amount_performance.idxmax()
        
        self.baseline_rules = {
            'default_best_success': best_success_psp,
            'cheapest_overall': cheapest_psp,
            'country_specific': country_best_psp,
            'amount_specific': amount_best_psp
        }
        
        print("Baseline Rules Created:")
        print(f"• Default high success PSP: {best_success_psp}")
        print(f"• Cheapest PSP: {cheapest_psp}")
        print(f"• Country-specific rules: {len(country_best_psp)} countries")
        print(f"• Amount-specific rules: {len(amount_best_psp)} categories")
        
        return self.baseline_rules
    
    def implement_baseline_strategy_1(self, df):
        """Strategy 1: Always use highest success rate PSP"""
        best_psp = self.baseline_rules['default_best_success']
        predictions = [best_psp] * len(df)
        return predictions
    
    def implement_baseline_strategy_2(self, df):
        """Strategy 2: Country-specific best PSP with cost consideration"""
        predictions = []
        for _, row in df.iterrows():
            country = row['country']
            # For small amounts, prioritize cost over success rate
            if row['amount'] < 100:
                predictions.append(self.baseline_rules['cheapest_overall'])
            elif country in self.baseline_rules['country_specific']:
                predictions.append(self.baseline_rules['country_specific'][country])
            else:
                predictions.append(self.baseline_rules['default_best_success'])
        return predictions
    
    def implement_baseline_strategy_3(self, df):
        """Strategy 3: Risk-based PSP selection"""
        predictions = []
        for _, row in df.iterrows():
            # High-risk transactions: use most reliable PSP regardless of cost
            if row['amount'] > 500 or row['card'] == 'Diners':
                predictions.append(self.baseline_rules['default_best_success'])
            # Medium-risk: balance cost and success
            elif row['amount'] > 200:
                # Use second-best PSP for cost optimization
                best_psps = self.historical_performance['overall']['Success_Rate'].nlargest(2)
                predictions.append(best_psps.index[1] if len(best_psps) > 1 else best_psps.index[0])
            # Low-risk: minimize cost
            else:
                predictions.append(self.baseline_rules['cheapest_overall'])
        return predictions
    
    def implement_baseline_strategy_4(self, df):
        """Strategy 4: Hybrid approach - cost-performance balance"""
        predictions = []
        for _, row in df.iterrows():
            # For high-value transactions, prioritize success rate
            if row['is_high_value'] == 1:
                predictions.append(self.baseline_rules['default_best_success'])
            # For low-value transactions, prioritize cost
            else:
                predictions.append(self.baseline_rules['cheapest_overall'])
        return predictions
    
    def evaluate_baseline_strategy(self, actual_data, predicted_psp, strategy_name):
        """Evaluate a baseline strategy"""
        
        # Create evaluation dataframe
        eval_df = actual_data.copy()
        eval_df['predicted_psp'] = predicted_psp
        
        # Validate predictions
        invalid_predictions = set(predicted_psp) - set(self.psp_fees.keys())
        if invalid_predictions:
            print(f"WARNING: Invalid PSP predictions found: {invalid_predictions}")
            # Replace invalid predictions with default
            eval_df['predicted_psp'] = eval_df['predicted_psp'].replace(
                invalid_predictions, self.baseline_rules['default_best_success']
            )
        
        # Get historical success rates for predicted PSPs
        psp_success_rates = self.historical_performance['overall']['Success_Rate'].to_dict()
        eval_df['predicted_success_prob'] = eval_df['predicted_psp'].map(psp_success_rates)
        
        # Handle any missing success rates
        if eval_df['predicted_success_prob'].isna().any():
            print(f"WARNING: Missing success rates for some PSPs, using overall average")
            overall_avg = self.historical_performance['overall']['Success_Rate'].mean()
            eval_df['predicted_success_prob'].fillna(overall_avg, inplace=True)
        
        # Simulate success based on historical rates (for baseline evaluation)
        np.random.seed(42)  # For reproducible results
        eval_df['predicted_success'] = np.random.binomial(1, eval_df['predicted_success_prob'])
        
        # Calculate costs
        eval_df['predicted_cost'] = eval_df.apply(
            lambda row: self.psp_fees[row['predicted_psp']]['success'] if row['predicted_success'] 
                       else self.psp_fees[row['predicted_psp']]['failure'], axis=1
        )
        
        # Current (actual) costs
        eval_df['actual_cost'] = eval_df.apply(
            lambda row: self.psp_fees[row['PSP']]['success'] if row['success'] 
                       else self.psp_fees[row['PSP']]['failure'], axis=1
        )
        
        # Calculate metrics
        predicted_success_rate = eval_df['predicted_success'].mean()
        actual_success_rate = eval_df['success'].mean()
        success_rate_improvement = predicted_success_rate - actual_success_rate
        
        predicted_avg_cost = eval_df['predicted_cost'].mean()
        actual_avg_cost = eval_df['actual_cost'].mean()
        cost_reduction = actual_avg_cost - predicted_avg_cost
        
        total_cost_savings = cost_reduction * len(eval_df)
        
        # Revenue impact (assuming 2.5% commission on successful transactions)
        commission_rate = 0.025
        predicted_revenue = eval_df[eval_df['predicted_success'] == 1]['amount'].sum() * commission_rate
        actual_revenue = eval_df[eval_df['success'] == 1]['amount'].sum() * commission_rate
        revenue_improvement = predicted_revenue - actual_revenue
        
        total_business_impact = total_cost_savings + revenue_improvement
        
        metrics = {
            'strategy': strategy_name,
            'predicted_success_rate': predicted_success_rate,
            'actual_success_rate': actual_success_rate,
            'success_rate_improvement': success_rate_improvement,
            'success_rate_improvement_pct': success_rate_improvement / actual_success_rate * 100,
            'predicted_avg_cost': predicted_avg_cost,
            'actual_avg_cost': actual_avg_cost,
            'cost_reduction': cost_reduction,
            'total_cost_savings': total_cost_savings,
            'revenue_improvement': revenue_improvement,
            'total_business_impact': total_business_impact
        }
        
        return metrics, eval_df
    
    def compare_baseline_strategies(self):
        """Compare all baseline strategies"""
        print("\n=== COMPARING BASELINE STRATEGIES ===")
        
        # Create test dataset (using historical data for simulation)
        test_data = self.df.sample(n=min(10000, len(self.df)), random_state=42)
        
        strategies = {
            'Strategy_1_Always_Best': self.implement_baseline_strategy_1,
            'Strategy_2_Cost_Aware': self.implement_baseline_strategy_2,
            'Strategy_3_Risk_Based': self.implement_baseline_strategy_3,
            'Strategy_4_Hybrid': self.implement_baseline_strategy_4
        }
        
        strategy_results = []
        
        for strategy_name, strategy_func in strategies.items():
            print(f"\nEvaluating {strategy_name}...")
            
            predictions = strategy_func(test_data)
            metrics, eval_df = self.evaluate_baseline_strategy(test_data, predictions, strategy_name)
            
            strategy_results.append(metrics)
            
            print(f"  Success Rate: {metrics['predicted_success_rate']:.1%} (vs {metrics['actual_success_rate']:.1%} current)")
            print(f"  Improvement: {metrics['success_rate_improvement_pct']:+.1f}%")
            print(f"  Avg Cost: €{metrics['predicted_avg_cost']:.2f} (vs €{metrics['actual_avg_cost']:.2f} current)")
            print(f"  Business Impact: €{metrics['total_business_impact']:,.0f}")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(strategy_results)
        
        # Find best strategy
        best_strategy = comparison_df.loc[comparison_df['total_business_impact'].idxmax()]
        
        print(f"\n=== BEST BASELINE STRATEGY ===")
        print(f"Winner: {best_strategy['strategy']}")
        print(f"Success Rate Improvement: {best_strategy['success_rate_improvement_pct']:+.1f}%")
        print(f"Total Business Impact: €{best_strategy['total_business_impact']:,.0f}")
        
        return comparison_df, best_strategy
    
    def visualize_baseline_results(self, comparison_df):
        """Create visualizations for baseline model results"""
        print("\n=== CREATING BASELINE VISUALIZATIONS ===")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Success Rate Comparison
        strategies = comparison_df['strategy']
        current_rate = comparison_df['actual_success_rate'].iloc[0]
        predicted_rates = comparison_df['predicted_success_rate']
        
        x_pos = np.arange(len(strategies))
        bars1 = ax1.bar(x_pos - 0.2, [current_rate] * len(strategies), 0.4, 
                       label='Current', color='red', alpha=0.7)
        bars2 = ax1.bar(x_pos + 0.2, predicted_rates, 0.4, 
                       label='Predicted', color='green', alpha=0.7)
        
        ax1.set_title('Success Rate Comparison', fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_xlabel('Strategy')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([s.replace('_', '\n') for s in strategies], rotation=0, ha='center')
        ax1.legend()
        ax1.set_ylim(0, max(predicted_rates) * 1.1)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Cost Comparison
        current_cost = comparison_df['actual_avg_cost'].iloc[0]
        predicted_costs = comparison_df['predicted_avg_cost']
        
        bars3 = ax2.bar(x_pos - 0.2, [current_cost] * len(strategies), 0.4,
                       label='Current', color='red', alpha=0.7)
        bars4 = ax2.bar(x_pos + 0.2, predicted_costs, 0.4,
                       label='Predicted', color='blue', alpha=0.7)
        
        ax2.set_title('Average Cost Comparison', fontweight='bold')
        ax2.set_ylabel('Average Cost (€)')
        ax2.set_xlabel('Strategy')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([s.replace('_', '\n') for s in strategies], rotation=0, ha='center')
        ax2.legend()
        
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'€{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Business Impact
        business_impacts = comparison_df['total_business_impact']
        colors = ['gold' if impact == business_impacts.max() else 'lightblue' for impact in business_impacts]
        
        bars5 = ax3.bar(strategies, business_impacts, color=colors)
        ax3.set_title('Total Business Impact', fontweight='bold')
        ax3.set_ylabel('Business Impact (€)')
        ax3.set_xlabel('Strategy')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars5:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                    f'€{height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Success Rate Improvement Percentage
        improvements = comparison_df['success_rate_improvement_pct']
        colors = ['gold' if imp == improvements.max() else 'lightgreen' for imp in improvements]
        
        bars6 = ax4.bar(strategies, improvements, color=colors)
        ax4.set_title('Success Rate Improvement', fontweight='bold')
        ax4.set_ylabel('Improvement (%)')
        ax4.set_xlabel('Strategy')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar in bars6:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:+.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/baseline_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_baseline_modeling(self):
        """Execute complete baseline modeling phase"""
        print("Starting CRISP-DM Phase 4a: Baseline Modeling")
        print("=" * 50)
        
        # Execute baseline modeling steps
        self.load_data()
        historical_performance = self.analyze_historical_performance()
        baseline_rules = self.create_baseline_rules()
        comparison_results, best_strategy = self.compare_baseline_strategies()
        self.visualize_baseline_results(comparison_results)
        
        # Compile baseline results
        baseline_results = {
            'historical_performance': historical_performance,
            'baseline_rules': baseline_rules,
            'strategy_comparison': comparison_results,
            'best_strategy': best_strategy,
            'modeling_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n=== BASELINE MODELING COMPLETE ===")
        print("Key achievements:")
        print(f"• {len(baseline_rules)} rule sets created")
        print(f"• {len(comparison_results)} strategies evaluated") 
        print(f"• Best strategy improves success rate by {best_strategy['success_rate_improvement_pct']:+.1f}%")
        print(f"• Potential business impact: €{best_strategy['total_business_impact']:,.0f}")
        
        return baseline_results

def main():
    """Main execution function"""
    baseline = BaselineModel()
    results = baseline.run_baseline_modeling()
    return results

if __name__ == "__main__":
    results = main() 