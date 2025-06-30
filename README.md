# Credit Card Routing Optimization - Academic Project

## Project Overview

This academic project addresses the business challenge of optimizing Payment Service Provider (PSP) routing for credit card transactions. The goal is to **increase success rates while minimizing transaction fees** through data-driven decision making.

## Assignment Requirements Compliance

Following the exact task requirements from the original assignment:

### ✅ 1. CRISP-DM Methodology & Git Repository Structure
Project structured following CRISP-DM phases (Business Understanding → Deployment)
Recommended git repository organization for production implementation
Clear separation between academic analysis and conceptual deployment

### ✅ 2. Data Quality Assessment & Business Visualization
Comprehensive data quality analysis with missing values, outliers, and consistency checks
Clear, business-stakeholder-friendly visualizations of initial data analysis findings
Executive-ready charts showing current PSP performance and optimization opportunities

### ✅ 3. Baseline Model + Accurate Predictive Model
**Baseline Model**: Simple rule-based approach for PSP selection
**Predictive Model**: Advanced ML model fulfilling business requirements
Both models designed to increase credit card success rate while keeping fees low

### ✅ 4. Model Interpretability & Error Analysis
Feature importance analysis with business interpretation
Model results explained in business terms for stakeholder confidence
Sophisticated error analysis identifying approach limitations and edge cases
Clear discussion of model drawbacks and mitigation strategies

## Data Overview

**Dataset**: 50,410 credit card transactions (Jan-Feb 2019, DACH region)
**PSPs Available**: Goldcard, Moneycard, UK_Card, Simplecard
**Current Success Rate**: ~20.3% (significant optimization opportunity)
**Business Challenge**: Balance success rate improvement with cost optimization

## Project Structure (CRISP-DM Methodology)

project_pnp/
├── input/
│   └── PSP_Jan_Feb_2019.csv          # Original transaction data
├── 01_business_understanding.py      # CRISP-DM Phase 1: Business problem analysis
├── 02_data_understanding.py          # CRISP-DM Phase 2: Data quality & EDA
├── 03_data_preparation.py            # CRISP-DM Phase 3: Feature engineering
├── 04_modeling_baseline.py           # CRISP-DM Phase 4a: Simple rule-based model
├── 05_modeling_predictive.py         # CRISP-DM Phase 4b: Advanced ML model
├── 06_evaluation.py                  # CRISP-DM Phase 5: Model comparison & validation
├── 07_deployment_proposal.py         # CRISP-DM Phase 6: Implementation recommendations
├── results/                          # Generated outputs
│   ├── data_quality_report.png
│   ├── business_stakeholder_charts.png
│   ├── baseline_model_results.png
│   ├── predictive_model_analysis.png
│   ├── feature_importance_analysis.png
│   ├── error_analysis_report.png
│   └── final_comparison_dashboard.png
└── backup_YYYYMMDD_HHMMSS/          # Previous work (reference)

## Recommended Git Repository Structure (Production)

psp-routing-optimization/
├── docs/                            # Documentation & reports
├── data/                            # Data storage (with .gitignore)
├── src/                             # Source code
│   ├── data_processing/            # ETL pipelines
│   ├── modeling/                   # ML models & training
│   ├── evaluation/                 # Testing & validation
│   └── deployment/                 # API & serving infrastructure
├── notebooks/                      # Jupyter analysis notebooks
├── config/                         # Configuration files
├── tests/                          # Unit & integration tests
└── requirements.txt                # Dependencies

## Key Business Insights (To Be Validated)

**Current State Issues:**
Goldcard (highest success 40.6%) severely underutilized at 6.4% volume
Heavy reliance on UK_Card (52.5% volume) with moderate success (19.4%)
Estimated annual opportunity: €10M+ in lost revenue and inefficient costs

**Expected Outcomes:**
**Baseline Model**: x% improvement through simple optimization rules
**Predictive Model**: y% improvement through ML-driven optimization
**Combined Impact**: Success rate improvement from m to n%+ achievable

## Implementation Approach (Following Assignment Requirements)

### Phase 1: CRISP-DM Business Understanding & Data Assessment
Define business problem: Optimize PSP routing to increase success rates and minimize fees
Assess data quality with comprehensive analysis (missing values, outliers, consistency)
Create business-stakeholder-friendly visualizations of current PSP performance

### Phase 2: Baseline Model Development
Implement simple rule-based PSP selection approach
Use historical success rates and basic business rules
Establish performance benchmark for comparison

### Phase 3: Predictive Model Development  
Develop advanced ML model using sophisticated feature engineering
Optimize for dual objective: success rate improvement + cost minimization
Implement proper model validation and hyperparameter tuning

### Phase 4: Model Interpretability & Error Analysis
Analyze feature importance with business interpretation
Provide sophisticated error analysis identifying model limitations
Create stakeholder-friendly explanations of model predictions
Document approach drawbacks and mitigation strategies

## Baseline Model Rules

The baseline model implements four distinct PSP routing strategies, each designed to test different business logic approaches for optimizing credit card transaction routing.

| Strategy | Decision Rules | PSPs Used | Thresholds | Business Logic | Expected Outcome |
|----------|----------------|-----------|------------|----------------|------------------|
| **1. Always Best** | Route all transactions to highest success rate PSP | Goldcard (40.6% success) | None | Simple success-rate maximization; conservative approach | Highest success rates, high transaction costs |
| **2. Cost-Country** | 1. Amount < €100 → Cheapest PSP<br>2. Country-specific best PSP<br>3. Fallback to best PSP | Simplecard (€1)<br>Country-specific<br>Goldcard (fallback) | €100 for cost optimization | Hierarchical: Cost → Geography → Success | Reduced costs for small transactions, maintained performance for large |
| **3. Risk-Based** | 1. >€500 OR Diners → Best PSP<br>2. €200-500 → Second-best PSP<br>3. <€200 → Cheapest PSP | Goldcard (high-risk)<br>Moneycard (medium-risk)<br>Simplecard (low-risk) | €200, €500<br>Diners card detection | 3-tier risk assessment with gradual cost-performance tradeoff | Balanced performance with risk-appropriate allocation |
| **4. Hybrid Value** | 1. Top 20% amounts → Best PSP<br>2. Bottom 80% → Cheapest PSP | Goldcard (high-value)<br>Simplecard (low-value) | 80th percentile (dynamic) | Binary high-stakes vs. low-stakes optimization | Clear cost savings for majority, protection for high-value |

### Strategy Comparison Matrix

| Dimension | Strategy 1 | Strategy 2 | Strategy 3 | Strategy 4 |
|-----------|------------|------------|------------|------------|
| **Complexity** | Simple | Medium | High | Medium |
| **Segmentation** | None | Geographic + Amount | Risk-based | Value-based |
| **PSP Variety** | 1 PSP | 2-3 PSPs | 3 PSPs | 2 PSPs |
| **Decision Criteria** | Success only | Cost + Geography | Risk + Amount + Card | Value percentile |
| **Special Cases** | None | Country rules | Diners cards | Dynamic threshold |
| **Optimization Focus** | Success-first | Cost-conscious | Risk-balanced | Binary optimization |

### Evaluation Framework

The baseline model evaluation compares these strategies across:
- **Success Rate Improvement**: How much each strategy improves transaction success
- **Cost Efficiency**: Average transaction fees and cost reduction achieved  
- **Business Impact**: Combined financial benefit from success + cost optimization
- **Implementation Complexity**: Practical considerations for business deployment

## Academic Success Criteria

**Data Quality**: Comprehensive assessment with stakeholder-ready visualizations
**Model Performance**: Baseline model + sophisticated predictive model both improving business metrics
**Interpretability**: Clear feature importance analysis and business explanations
**Error Analysis**: Sophisticated analysis of model limitations and edge cases
**CRISP-DM Compliance**: Full methodology implementation with git repository recommendation

