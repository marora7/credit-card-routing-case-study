"""
CRISP-DM Phase 1: Business Understanding
Credit Card Routing Optimization Project

This script defines the business problem, objectives, and success criteria
for optimizing PSP routing to increase success rates and minimize fees.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BusinessUnderstanding:
    """Business Understanding Phase - Define problem and objectives"""
    
    def __init__(self):
        self.psp_fees = {
            'Moneycard': {'success': 5.00, 'failure': 2.00},
            'Goldcard': {'success': 10.00, 'failure': 5.00},
            'UK_Card': {'success': 3.00, 'failure': 1.00},
            'Simplecard': {'success': 1.00, 'failure': 0.50}
        }
        
        self.business_objectives = {
            'primary': 'Increase credit card payment success rate',
            'secondary': 'Minimize transaction fees',
            'constraint': 'Maintain customer satisfaction'
        }
        
    def define_business_problem(self):
        """Define the core business problem"""
        problem_statement = {
            'current_situation': 'High failure rate of online credit card payments',
            'business_impact': 'Lost revenue and decreased customer satisfaction',
            'current_approach': 'Manual rule-based PSP routing',
            'desired_outcome': 'Automated intelligent PSP routing',
            'success_metrics': ['Payment success rate increase', 'Cost reduction', 'Customer satisfaction']
        }
        
        print("=== BUSINESS PROBLEM DEFINITION ===")
        for key, value in problem_statement.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        return problem_statement
    
    def analyze_psp_costs(self):
        """Analyze PSP cost structure"""
        print("\n=== PSP COST ANALYSIS ===")
        
        # Create PSP cost comparison
        psp_names = list(self.psp_fees.keys())
        success_fees = [self.psp_fees[psp]['success'] for psp in psp_names]
        failure_fees = [self.psp_fees[psp]['failure'] for psp in psp_names]
        
        # Create cost comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Success fees
        bars1 = ax1.bar(psp_names, success_fees, color='green', alpha=0.7)
        ax1.set_title('Success Fees by PSP')
        ax1.set_ylabel('Fee (€)')
        ax1.set_ylim(0, max(success_fees) * 1.1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'€{height:.2f}', ha='center', va='bottom')
        
        # Failure fees
        bars2 = ax2.bar(psp_names, failure_fees, color='red', alpha=0.7)
        ax2.set_title('Failure Fees by PSP')
        ax2.set_ylabel('Fee (€)')
        ax2.set_ylim(0, max(failure_fees) * 1.1)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'€{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/psp_cost_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.psp_fees
    
    def calculate_potential_savings(self, total_transactions=50000, current_success_rate=0.203):
        """Calculate potential business impact"""
        print(f"\n=== POTENTIAL BUSINESS IMPACT ===")
        
        # Assumptions
        avg_transaction_value = 100  # €100 average transaction
        current_failure_rate = 1 - current_success_rate
        
        # Current costs (assuming even distribution across PSPs)
        avg_success_fee = np.mean([psp['success'] for psp in self.psp_fees.values()])
        avg_failure_fee = np.mean([psp['failure'] for psp in self.psp_fees.values()])
        
        current_annual_fees = (total_transactions * current_success_rate * avg_success_fee + 
                              total_transactions * current_failure_rate * avg_failure_fee)
        
        # Lost revenue from failed transactions
        current_lost_revenue = total_transactions * current_failure_rate * avg_transaction_value
        
        # Potential improvements (conservative estimates)
        target_success_rate = min(0.35, current_success_rate + 0.15)  # 15% improvement target
        optimized_success_rate = target_success_rate
        optimized_failure_rate = 1 - optimized_success_rate
        
        # Optimized costs (using cheaper PSPs more effectively)
        optimized_avg_success_fee = 2.5  # Weighted towards cheaper PSPs
        optimized_avg_failure_fee = 1.2
        
        optimized_annual_fees = (total_transactions * optimized_success_rate * optimized_avg_success_fee + 
                               total_transactions * optimized_failure_rate * optimized_avg_failure_fee)
        
        optimized_lost_revenue = total_transactions * optimized_failure_rate * avg_transaction_value
        
        # Calculate savings
        fee_savings = current_annual_fees - optimized_annual_fees
        revenue_recovery = current_lost_revenue - optimized_lost_revenue
        total_impact = fee_savings + revenue_recovery
        
        print(f"Current Success Rate: {current_success_rate:.1%}")
        print(f"Target Success Rate: {optimized_success_rate:.1%}")
        print(f"Current Annual Fees: €{current_annual_fees:,.0f}")
        print(f"Optimized Annual Fees: €{optimized_annual_fees:,.0f}")
        print(f"Fee Savings: €{fee_savings:,.0f}")
        print(f"Revenue Recovery: €{revenue_recovery:,.0f}")
        print(f"Total Annual Impact: €{total_impact:,.0f}")
        
        return {
            'current_success_rate': current_success_rate,
            'target_success_rate': optimized_success_rate,
            'fee_savings': fee_savings,
            'revenue_recovery': revenue_recovery,
            'total_impact': total_impact
        }
    
    def define_success_criteria(self):
        """Define project success criteria"""
        success_criteria = {
            'primary_kpi': 'Payment success rate improvement >= 10%',
            'secondary_kpi': 'Transaction cost reduction >= 15%',
            'model_performance': 'Model accuracy >= 80%',
            'business_impact': 'Annual savings >= €1M',
            'implementation': 'Model interpretability for business confidence'
        }
        
        print(f"\n=== PROJECT SUCCESS CRITERIA ===")
        for criterion, target in success_criteria.items():
            print(f"{criterion.replace('_', ' ').title()}: {target}")
        
        return success_criteria
    
    def run_business_understanding(self):
        """Execute complete business understanding phase"""
        print("Starting CRISP-DM Phase 1: Business Understanding")
        print("=" * 50)
        
        # Execute all business understanding tasks
        problem = self.define_business_problem()
        costs = self.analyze_psp_costs()
        impact = self.calculate_potential_savings()
        criteria = self.define_success_criteria()
        
        # Save business understanding summary
        summary = {
            'problem_definition': problem,
            'psp_costs': costs,
            'business_impact': impact,
            'success_criteria': criteria,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        print(f"\n=== BUSINESS UNDERSTANDING COMPLETE ===")
        print("Key outputs saved to results/ directory")
        
        return summary

def main():
    """Main execution function"""
    business = BusinessUnderstanding()
    summary = business.run_business_understanding()
    return summary

if __name__ == "__main__":
    summary = main() 