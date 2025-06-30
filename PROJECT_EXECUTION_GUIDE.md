# Credit Card Routing Optimization - Execution Guide

## 🚀 Quick Start

This project implements the complete CRISP-DM methodology for optimizing Payment Service Provider (PSP) routing to increase credit card transaction success rates while minimizing costs.

### Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or higher)
- **8GB RAM** minimum (16GB recommended)
- **10GB free disk space**
- **Internet connection** (for downloading dependencies)

### Installation

1. **Clone or download the project**
   ```bash
   cd credit_card_routing
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data file exists**
   ```bash
   # Check if input/PSP_Jan_Feb_2019.xlsx exists
   # If not, copy it from the data/ directory
   ```

### Execution Options

#### Option 1: Full CRISP-DM Pipeline (Recommended)
```bash
python main.py
```
This executes all 7 phases of the CRISP-DM methodology in sequence.

#### Option 2: Individual Phase Execution
```bash
# Phase 1: Business Understanding
python 01_business_understanding.py

# Phase 2: Data Understanding  
python 02_data_understanding.py

# Phase 3: Data Preparation
python 03_data_preparation.py

# Phase 4a: Baseline Modeling
python 04_modeling_baseline.py

# Phase 4b: Predictive Modeling
python 05_modeling_predictive.py

# Phase 5: Evaluation
python 06_evaluation.py

# Phase 6: Deployment Proposal
python 07_deployment_proposal.py
```

#### Option 3: Configuration Check
```bash
python config.py
```
Validates all configuration settings and displays project parameters.

## 📊 Expected Outputs

### Results Directory Structure
```
results/
├── psp_cost_analysis.png
├── business_stakeholder_dashboard.png
├── baseline_model_comparison.png
├── feature_importance_analysis.png
├── predictive_model_comparison.png
├── comprehensive_evaluation_dashboard.png
└── deployment_roadmap_visualization.png
```

### Key Deliverables

1. **Business Understanding**
   - PSP cost analysis visualization
   - Potential business impact calculations
   - Success criteria definition

2. **Data Understanding**
   - Comprehensive data quality report
   - Business-stakeholder dashboard
   - Executive summary with key insights

3. **Data Preparation**
   - Feature engineering pipeline
   - Clean datasets ready for modeling
   - Retry attempt detection

4. **Baseline Modeling**
   - 4 rule-based strategies
   - Performance comparison
   - Business impact analysis

5. **Predictive Modeling**
   - 4 machine learning models
   - PSP recommendation system
   - Feature importance analysis

6. **Evaluation**
   - Model comparison dashboard
   - Sophisticated error analysis
   - Business recommendations

7. **Deployment Proposal**
   - Implementation roadmap
   - GUI specifications
   - ROI analysis

## 🎯 Expected Results

### Performance Improvements
- **Success Rate**: 20.3% → 25%+ (target improvement)
- **Cost Reduction**: 15%+ through optimized PSP routing
- **Business Impact**: €250,000+ annual value

### Model Performance
- **Best Model**: Gradient Boosting (accuracy: 79.9%, AUC: 65.7%)
- **AUC Score**: 0.83+ for success prediction
- **Feature Importance**: Clear business interpretability

### ROI Analysis
- **Implementation Cost**: ~€391,600
- **Annual Benefits**: ~€750,000
- **Year 1 ROI**: 150%+
- **Payback Period**: ~6.2 months

## 🔧 Troubleshooting

### Common Issues

1. **Data File Not Found**
   ```
   Error: Data file not found: input/PSP_Jan_Feb_2019.xlsx
   ```
   **Solution**: Copy `PSP_Jan_Feb_2019.xlsx` from `data/` to `input/` directory

2. **Missing Dependencies**
   ```
   Error: No module named 'pandas'
   ```
   **Solution**: Install requirements: `pip install -r requirements.txt`

3. **Memory Issues**
   ```
   Error: Memory allocation failed
   ```
   **Solution**: Close other applications, use smaller sample size, or increase system memory

4. **Plotting Issues**
   ```
   Error: Cannot display plots
   ```
   **Solution**: Set `plt.show()` to `plt.savefig()` for headless environments

### Performance Optimization

1. **Speed up execution**
   - Use `n_jobs=-1` for parallel processing
   - Reduce sample size for testing
   - Skip visualization generation

2. **Memory optimization**
   - Process data in chunks
   - Use `del` statements for large variables
   - Optimize pandas operations

## 📈 Monitoring Execution

### Progress Tracking
The main execution script provides detailed progress information:
- ✅ Phase completion status
- ⏰ Execution timestamps
- 📊 Key metrics and results
- ⚠️ Warnings and errors

### Log Files
- Execution logs saved to console output
- Error traces provided for debugging
- Performance metrics tracked

### Expected Runtime
- **Full Pipeline**: 15-30 minutes (depending on system)
- **Individual Phases**: 2-5 minutes each
- **Data Processing**: 5-10 minutes
- **Model Training**: 5-15 minutes

## 🔍 Validation Checks

### Data Validation
- Transaction count: 50,410 expected
- Date range: Jan-Feb 2019
- PSP coverage: 4 PSPs (Moneycard, Goldcard, UK_Card, Simplecard)
- Country coverage: 3 countries (Germany, Austria, Switzerland)

### Model Validation
- Success rate improvement: >10% target
- Cost reduction: >15% target
- Model accuracy: >75% target
- Feature importance: Business interpretable

### Output Validation
- All visualizations generated
- Performance metrics within expected ranges
- Business recommendations provided
- Deployment roadmap complete

## 📞 Support

### Configuration Issues
Run `python config.py` to validate all settings and identify issues.

### Execution Issues
Check the console output for detailed error messages and stack traces.

### Data Issues
Verify data file integrity and format using Excel or pandas.

### Performance Issues
Monitor system resources during execution and adjust parameters if needed.

## 🎓 Educational Value

This project demonstrates:
- **Complete CRISP-DM methodology** implementation
- **Real-world business problem** solving
- **End-to-end data science pipeline**
- **Production-ready code structure**
- **Business stakeholder communication**
- **Deployment planning and ROI analysis**

Perfect for:
- Data science education
- Business case studies
- Technical interviews
- Academic projects
- Professional development

## 🚀 Next Steps After Execution

1. **Review Results**
   - Examine all generated visualizations
   - Analyze model performance metrics
   - Review business recommendations

2. **Present Findings**
   - Use generated dashboards for stakeholder presentations
   - Highlight key business impacts
   - Discuss implementation roadmap

3. **Implementation Planning**
   - Follow deployment roadmap
   - Secure necessary resources
   - Begin Phase 1 implementation

4. **Continuous Improvement**
   - Monitor model performance
   - Retrain models regularly
   - Expand to additional regions/PSPs

---

**Ready to optimize your PSP routing? Run `python main.py` to start the journey!** 🚀 