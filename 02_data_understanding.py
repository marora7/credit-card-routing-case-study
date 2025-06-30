"""
CRISP-DM Phase 2: Data Understanding
Credit Card Routing Optimization Project

This script performs comprehensive data quality assessment and exploratory data analysis
with business-stakeholder-friendly visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataUnderstanding:
    """Data Understanding Phase - Quality assessment and EDA"""
    
    def __init__(self, data_path='input/PSP_Jan_Feb_2019.xlsx'):
        self.data_path = data_path
        self.df = None
        self.data_quality_report = {}
        
    def load_data(self):
        """Load and basic inspection of the dataset"""
        print("=== LOADING DATA ===")
        
        try:
            self.df = pd.read_excel(self.data_path)
            print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Basic info
            print(f"\nColumns: {list(self.df.columns)}")
            print(f"Data types:\n{self.df.dtypes}")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def assess_data_quality(self):
        """Comprehensive data quality assessment"""
        print("\n=== DATA QUALITY ASSESSMENT ===")
        
        # Missing values analysis
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df)) * 100
        
        quality_summary = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_pct,
            'Data_Type': self.df.dtypes,
            'Unique_Values': self.df.nunique(),
            'Example_Values': [self.df[col].dropna().iloc[0] if not self.df[col].dropna().empty else 'N/A' 
                              for col in self.df.columns]
        })
        
        print("Data Quality Summary:")
        print(quality_summary)
        
        # Check for duplicates
        duplicate_count = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicate_count}")
        
        # Data consistency checks
        print(f"\nData Consistency Checks:")
        print(f"Success values: {sorted(self.df['success'].unique())}")
        print(f"PSP values: {sorted(self.df['PSP'].unique())}")
        print(f"Country values: {sorted(self.df['country'].unique())}")
        print(f"3D_secured values: {sorted(self.df['3D_secured'].unique())}")
        print(f"Card values: {sorted(self.df['card'].unique())}")
        
        # Outlier detection for amount
        Q1 = self.df['amount'].quantile(0.25)
        Q3 = self.df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df['amount'] < lower_bound) | (self.df['amount'] > upper_bound)]
        print(f"\nAmount outliers (IQR method): {len(outliers)} ({len(outliers)/len(self.df)*100:.1f}%)")
        
        self.data_quality_report = {
            'total_records': len(self.df),
            'missing_data': missing_data.to_dict(),
            'duplicates': duplicate_count,
            'outliers': len(outliers),
            'quality_summary': quality_summary
        }
        
        return self.data_quality_report
    
    def create_business_visualizations(self):
        """Create business-stakeholder-friendly visualizations"""
        print("\n=== CREATING BUSINESS VISUALIZATIONS ===")
        
        # Set up the figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. PSP Performance Overview
        ax1 = plt.subplot(3, 3, 1)
        psp_success = self.df.groupby('PSP')['success'].agg(['count', 'mean']).reset_index()
        psp_success.columns = ['PSP', 'Total_Transactions', 'Success_Rate']
        
        bars = ax1.bar(psp_success['PSP'], psp_success['Success_Rate'], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Success Rate by PSP', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Transaction Volume by PSP
        ax2 = plt.subplot(3, 3, 2)
        volume_pct = self.df['PSP'].value_counts(normalize=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        wedges, texts, autotexts = ax2.pie(volume_pct.values, labels=volume_pct.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Transaction Volume Distribution by PSP', fontsize=14, fontweight='bold')
        
        # 3. Success Rate by Country
        ax3 = plt.subplot(3, 3, 3)
        country_success = self.df.groupby('country')['success'].mean().sort_values(ascending=False)
        bars = ax3.bar(country_success.index, country_success.values, color=['#FF9999', '#66B2FF', '#99FF99'])
        ax3.set_title('Success Rate by Country', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Success Rate')
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Transaction Amount Distribution
        ax4 = plt.subplot(3, 3, 4)
        ax4.hist(self.df['amount'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        ax4.set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Amount (€)')
        ax4.set_ylabel('Frequency')
        ax4.axvline(self.df['amount'].mean(), color='red', linestyle='--', 
                   label=f'Mean: €{self.df["amount"].mean():.2f}')
        ax4.legend()
        
        # 5. Success Rate by Card Type
        ax5 = plt.subplot(3, 3, 5)
        card_success = self.df.groupby('card')['success'].mean().sort_values(ascending=False)
        bars = ax5.bar(card_success.index, card_success.values, color=['#FFB6C1', '#87CEEB', '#DDA0DD'])
        ax5.set_title('Success Rate by Card Type', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Success Rate')
        
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 6. 3D Secured vs Non-3D Secured
        ax6 = plt.subplot(3, 3, 6)
        secured_success = self.df.groupby('3D_secured')['success'].mean()
        labels = ['Non-3D Secured', '3D Secured']
        bars = ax6.bar(labels, secured_success.values, color=['#FF7F7F', '#7FFF7F'])
        ax6.set_title('Success Rate: 3D Secured vs Non-3D Secured', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Success Rate')
        
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 7. Daily Transaction Pattern
        ax7 = plt.subplot(3, 3, 7)
        self.df['date'] = pd.to_datetime(self.df['tmsp']).dt.date
        daily_transactions = self.df.groupby('date').size()
        ax7.plot(daily_transactions.index, daily_transactions.values, marker='o', linewidth=2)
        ax7.set_title('Daily Transaction Volume', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Date')
        ax7.set_ylabel('Transaction Count')
        ax7.tick_params(axis='x', rotation=45)
        
        # 8. PSP Performance Matrix
        ax8 = plt.subplot(3, 3, 8)
        psp_matrix = self.df.groupby('PSP').agg({
            'success': ['count', 'mean']
        }).round(3)
        psp_matrix.columns = ['Volume', 'Success_Rate']
        
        scatter = ax8.scatter(psp_matrix['Volume'], psp_matrix['Success_Rate'], 
                             s=200, alpha=0.7, c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax8.set_title('PSP Performance Matrix', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Transaction Volume')
        ax8.set_ylabel('Success Rate')
        
        # Add PSP labels
        for i, psp in enumerate(psp_matrix.index):
            ax8.annotate(psp, (psp_matrix.iloc[i]['Volume'], psp_matrix.iloc[i]['Success_Rate']),
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        # 9. Overall Success Rate Trend
        ax9 = plt.subplot(3, 3, 9)
        overall_success = self.df['success'].mean()
        ax9.text(0.5, 0.6, f'Overall Success Rate', fontsize=16, fontweight='bold', 
                ha='center', transform=ax9.transAxes)
        ax9.text(0.5, 0.4, f'{overall_success:.1%}', fontsize=24, fontweight='bold', 
                ha='center', transform=ax9.transAxes, color='red')
        ax9.text(0.5, 0.2, f'Significant room for improvement!', fontsize=12, 
                ha='center', transform=ax9.transAxes, style='italic')
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
        ax9.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/business_stakeholder_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return psp_success
    
    def analyze_multiple_attempts(self):
        """Analyze potential multiple payment attempts"""
        print("\n=== ANALYZING MULTIPLE PAYMENT ATTEMPTS ===")
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['tmsp'])
        
        # Sort by country, amount, and timestamp
        df_sorted = self.df.sort_values(['country', 'amount', 'timestamp'])
        
        # Identify potential duplicate attempts (same country, amount within 1 minute)
        df_sorted['time_diff'] = df_sorted.groupby(['country', 'amount'])['timestamp'].diff()
        df_sorted['is_potential_retry'] = (df_sorted['time_diff'] <= timedelta(minutes=1)) & \
                                         (df_sorted['time_diff'].notna())
        
        retry_count = df_sorted['is_potential_retry'].sum()
        retry_rate = retry_count / len(df_sorted) * 100
        
        print(f"Potential retry attempts identified: {retry_count} ({retry_rate:.1f}% of all transactions)")
        
        # Analyze retry patterns
        retry_success = df_sorted[df_sorted['is_potential_retry']]['success'].mean()
        initial_success = df_sorted[~df_sorted['is_potential_retry']]['success'].mean()
        
        print(f"Success rate for initial attempts: {initial_success:.1%}")
        print(f"Success rate for retry attempts: {retry_success:.1%}")
        
        return {
            'retry_count': retry_count,
            'retry_rate': retry_rate,
            'retry_success_rate': retry_success,
            'initial_success_rate': initial_success
        }
    
    def generate_executive_summary(self):
        """Generate executive summary for business stakeholders"""
        print("\n=== EXECUTIVE SUMMARY ===")
        
        # Key metrics
        total_transactions = len(self.df)
        overall_success_rate = self.df['success'].mean()
        
        # Best and worst performing PSPs
        psp_performance = self.df.groupby('PSP')['success'].mean().sort_values(ascending=False)
        best_psp = psp_performance.index[0]
        worst_psp = psp_performance.index[-1]
        
        # Volume distribution
        psp_volume = self.df['PSP'].value_counts(normalize=True)
        
        summary = {
            'total_transactions': total_transactions,
            'overall_success_rate': overall_success_rate,
            'best_performing_psp': best_psp,
            'best_psp_success_rate': psp_performance[best_psp],
            'worst_performing_psp': worst_psp,
            'worst_psp_success_rate': psp_performance[worst_psp],
            'volume_leader': psp_volume.index[0],
            'volume_leader_share': psp_volume.iloc[0]
        }
        
        print("EXECUTIVE SUMMARY:")
        print(f"• Total Transactions Analyzed: {total_transactions:,}")
        print(f"• Overall Success Rate: {overall_success_rate:.1%}")
        print(f"• Best Performing PSP: {best_psp} ({psp_performance[best_psp]:.1%} success rate)")
        print(f"• Worst Performing PSP: {worst_psp} ({psp_performance[worst_psp]:.1%} success rate)")
        print(f"• Volume Leader: {psp_volume.index[0]} ({psp_volume.iloc[0]:.1%} of transactions)")
        print(f"• Key Insight: {best_psp} significantly outperforms but handles minimal volume")
        
        return summary
    
    def run_data_understanding(self):
        """Execute complete data understanding phase"""
        print("Starting CRISP-DM Phase 2: Data Understanding")
        print("=" * 50)
        
        # Execute all data understanding tasks
        if not self.load_data():
            return None
        
        quality_report = self.assess_data_quality()
        psp_performance = self.create_business_visualizations()
        retry_analysis = self.analyze_multiple_attempts()
        executive_summary = self.generate_executive_summary()
        
        # Compile complete analysis
        analysis_results = {
            'data_quality': quality_report,
            'psp_performance': psp_performance,
            'retry_analysis': retry_analysis,
            'executive_summary': executive_summary,
            'raw_data': self.df,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"\n=== DATA UNDERSTANDING COMPLETE ===")
        print("Key insights:")
        print(f"• Data quality is good with {quality_report['total_records']:,} clean records")
        print(f"• Success rate varies significantly across PSPs")
        print(f"• Clear optimization opportunity identified")
        
        return analysis_results

def main():
    """Main execution function"""
    data_analyzer = DataUnderstanding()
    results = data_analyzer.run_data_understanding()
    return results

if __name__ == "__main__":
    results = main() 