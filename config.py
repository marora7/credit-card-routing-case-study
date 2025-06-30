"""
Configuration File - Credit Card Routing Optimization Project

This file contains all project configuration settings, parameters,
and constants used across the CRISP-DM methodology implementation.
"""

import os
from datetime import datetime

# Project Information
PROJECT_NAME = "Credit Card Routing Optimization"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "CRISP-DM methodology implementation for PSP routing optimization"
AUTHOR = "Data Science Team"
CREATED_DATE = "2024-01-01"

# File Paths
DATA_PATH = "input/PSP_Jan_Feb_2019.xlsx"
RESULTS_PATH = "results/"
BACKUP_PATH = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"

# PSP Configuration
PSP_FEES = {
    'Moneycard': {'success': 5.00, 'failure': 2.00},
    'Goldcard': {'success': 10.00, 'failure': 5.00},
    'UK_Card': {'success': 3.00, 'failure': 1.00},
    'Simplecard': {'success': 1.00, 'failure': 0.50}
}

PSP_NAMES = list(PSP_FEES.keys())

# Business Parameters
BUSINESS_OBJECTIVES = {
    'primary': 'Increase credit card payment success rate',
    'secondary': 'Minimize transaction fees',
    'constraint': 'Maintain customer satisfaction'
}

SUCCESS_RATE_TARGET = 0.25  # 25% target success rate
COST_REDUCTION_TARGET = 0.15  # 15% cost reduction target
BUSINESS_IMPACT_TARGET = 250000  # €250K annual impact target

# Model Configuration
RANDOM_STATE = 42  # For reproducible results
TEST_SIZE = 0.2  # 20% for testing
CROSS_VALIDATION_FOLDS = 5

# Feature Engineering Parameters
HIGH_VALUE_PERCENTILE = 0.8  # 80th percentile for high-value transactions
MICRO_TRANSACTION_PERCENTILE = 0.1  # 10th percentile for micro transactions
RETRY_DETECTION_WINDOW = 5  # 5 minutes for retry detection

# Model Parameters
BASELINE_MODELS = {
    'strategy_1': 'Always use highest success rate PSP',
    'strategy_2': 'Country-specific best PSP',
    'strategy_3': 'Amount-based PSP selection',
    'strategy_4': 'Hybrid cost-performance balance'
}

PREDICTIVE_MODELS = {
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    },
    'random_forest': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE
    },
    'xgboost': {
        'n_estimators': 100,
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss'
    }
}

# Visualization Settings
FIGURE_SIZE = (12, 8)
FIGURE_DPI = 300
COLOR_PALETTE = "husl"
PLOT_STYLE = "seaborn-v0_8"

# Dashboard Configuration
DASHBOARD_REFRESH_RATES = {
    'overview': 30,  # seconds
    'psp_management': 60,  # seconds
    'model_monitoring': 3600,  # 1 hour
    'transactions': 60,  # seconds
    'configuration': 'on_demand'
}

# Alert Thresholds
ALERT_THRESHOLDS = {
    'success_rate_drop': 0.05,  # 5% drop triggers alert
    'cost_increase': 0.10,  # 10% cost increase triggers alert
    'model_accuracy_drop': 0.03,  # 3% accuracy drop triggers alert
    'transaction_volume_spike': 2.0  # 2x normal volume triggers alert
}

# Deployment Configuration
DEPLOYMENT_PHASES = {
    'phase_1': {'duration_weeks': 6, 'name': 'Foundation'},
    'phase_2': {'duration_weeks': 8, 'name': 'ML Integration'},
    'phase_3': {'duration_weeks': 6, 'name': 'Optimization'},
    'phase_4': {'duration_weeks': 8, 'name': 'Enhancement'}
}

# ROI Configuration
IMPLEMENTATION_COSTS = {
    'development': 294000,  # €294K
    'infrastructure': 39600,  # €39.6K (12 months)
    'one_time': 58000  # €58K
}

ANNUAL_BENEFITS = {
    'increased_revenue': 450000,  # €450K
    'cost_savings': 180000,  # €180K
    'operational_efficiency': 75000,  # €75K
    'reduced_manual_effort': 45000  # €45K
}

# System Configuration
SYSTEM_REQUIREMENTS = {
    'python_version': '>=3.8',
    'memory_gb': 8,
    'storage_gb': 10,
    'cpu_cores': 4
}

# API Configuration
API_ENDPOINTS = {
    'routing': '/api/v1/route-psp',
    'performance': '/api/v1/performance',
    'configuration': '/api/v1/config'
}

# Security Configuration
SECURITY_SETTINGS = {
    'api_authentication': True,
    'data_encryption': True,
    'audit_logging': True,
    'role_based_access': True
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/project.log'
}

def get_config_summary():
    """Get a summary of all configuration settings"""
    return {
        'project_info': {
            'name': PROJECT_NAME,
            'version': PROJECT_VERSION,
            'description': PROJECT_DESCRIPTION
        },
        'data_config': {
            'data_path': DATA_PATH,
            'results_path': RESULTS_PATH,
            'test_size': TEST_SIZE
        },
        'business_config': {
            'success_rate_target': SUCCESS_RATE_TARGET,
            'cost_reduction_target': COST_REDUCTION_TARGET,
            'business_impact_target': BUSINESS_IMPACT_TARGET
        },
        'model_config': {
            'random_state': RANDOM_STATE,
            'cv_folds': CROSS_VALIDATION_FOLDS,
            'baseline_models': len(BASELINE_MODELS),
            'predictive_models': len(PREDICTIVE_MODELS)
        },
        'deployment_config': {
            'total_phases': len(DEPLOYMENT_PHASES),
            'total_duration_weeks': sum(phase['duration_weeks'] for phase in DEPLOYMENT_PHASES.values())
        }
    }

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        errors.append(f"Data file not found: {DATA_PATH}")
    
    # Check if results directory exists
    if not os.path.exists(RESULTS_PATH):
        try:
            os.makedirs(RESULTS_PATH)
        except Exception as e:
            errors.append(f"Cannot create results directory: {e}")
    
    # Validate business parameters
    if not (0 < SUCCESS_RATE_TARGET < 1):
        errors.append("Success rate target must be between 0 and 1")
    
    if not (0 < COST_REDUCTION_TARGET < 1):
        errors.append("Cost reduction target must be between 0 and 1")
    
    # Validate model parameters
    if not (0 < TEST_SIZE < 1):
        errors.append("Test size must be between 0 and 1")
    
    if CROSS_VALIDATION_FOLDS < 2:
        errors.append("Cross-validation folds must be at least 2")
    
    return errors

if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("CREDIT CARD ROUTING OPTIMIZATION - CONFIGURATION")
    print("=" * 60)
    
    config_summary = get_config_summary()
    
    for section, settings in config_summary.items():
        print(f"\n{section.replace('_', ' ').title()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    # Validate configuration
    print(f"\n{'='*60}")
    print("CONFIGURATION VALIDATION")
    print("=" * 60)
    
    errors = validate_config()
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"  • {error}")
    else:
        print("✅ Configuration validation passed")
    
    print(f"\n{'='*60}")
    print("Ready to execute CRISP-DM methodology!") 