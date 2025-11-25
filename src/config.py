import os

# 1. PROJECT PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')


# 2. GLOBAL SETTINGS & REPRODUCIBILITY
RANDOM_SEED = 42
N_JOBS = -1  # Use all available CPU cores

# Data Splitting Strategy
TRAIN_SIZE = 0.70
TEST_SIZE = 0.20
VAL_SIZE = 0.10

# Cross-Validation Settings (For Deep Search)
CV_FOLDS = 5

# Inhibition Thresholds
THRESHOLD_50 = 50.0
THRESHOLD_20 = 20.0


# 3. DEEP HYPERPARAMETER GRIDS - REGRESSION (Binding Affinity)

# --- A. XGBoost Regression (Deep Grid) ---
XGB_REG_GRID = {
    'n_estimators': [100, 200, 300, 500, 1000, 1500],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    'min_child_weight': [1, 2, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.001, 0.01, 0.1, 1, 10],  # L1 Regularization
    'reg_lambda': [0, 0.001, 0.01, 0.1, 1, 10], # L2 Regularization
    'booster': ['gbtree', 'gblinear', 'dart']
}

# --- B. CatBoost Regression (Deep Grid) ---
CATBOOST_REG_GRID = {
    'iterations': [500, 1000, 2000, 3000, 5000],
    'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1],
    'depth': [4, 6, 8, 10, 12],
    'l2_leaf_reg': [1, 3, 5, 7, 9, 15, 20],
    'border_count': [32, 64, 128, 254],
    'bagging_temperature': [0, 0.2, 0.5, 0.8, 1],
    'random_strength': [1, 5, 10],
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
}

# --- C. LightGBM Regression (Deep Grid) ---
LIGHTGBM_REG_GRID = {
    'n_estimators': [100, 500, 1000, 2000],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 63, 127, 255], # Important: < 2^max_depth
    'max_depth': [-1, 5, 10, 15, 20, 30],
    'min_child_samples': [5, 10, 20, 50, 100],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'subsample_freq': [0, 1, 5],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0.0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.0, 0.1, 0.5, 1.0],
    'boosting_type': ['gbdt', 'dart', 'goss']
}

# --- D. Random Forest Regression (Deep Grid) ---
RF_REG_GRID = {
    'n_estimators': [100, 200, 300, 500, 800, 1000],
    'max_depth': [None, 10, 20, 30, 40, 50, 80, 100],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 8, 10],
    'max_features': ['sqrt', 'log2', None, 0.5], # 0.5 means 50% features
    'bootstrap': [True, False],
    'criterion': ['squared_error', 'absolute_error', 'poisson']
}

# --- E. MLP Regression (Deep Grid) ---
MLP_REG_GRID = {
    'hidden_layer_sizes': [
        (50,), (100,), (200,),              # Shallow
        (50, 50), (100, 50), (100, 100),    # Medium
        (100, 50, 25), (200, 100, 50),      # Deep Funnel
        (50, 50, 50, 50),                   # Very Deep
        (1024, 512, 256, 128)               # Massive
    ],
    'activation': ['relu', 'tanh', 'logistic', 'identity'],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0], # L2 Regularization
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'batch_size': ['auto', 32, 64, 128],
    'momentum': [0.9, 0.95, 0.99]
}


# =============================================================================
# 4. DEEP HYPERPARAMETER GRIDS - CLASSIFICATION (Inhibition)
# =============================================================================

# --- A. XGBoost Classification (Deep Grid) ---
XGB_CLS_GRID = {
    'n_estimators': [100, 300, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [1, 2, 5, 10, 20], # CRITICAL for Imbalance
    'reg_alpha': [0.001, 0.005, 0.01, 0.1],
    'reg_lambda': [0.1, 1.0, 5.0]
}

# --- B. CatBoost Classification (Deep Grid) ---
CATBOOST_CLS_GRID = {
    'iterations': [50, 100, 200, 500, 1000], # Varied for overfitting check
    'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
    'depth': [4, 5, 6, 7, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9, 12],
    'border_count': [32, 128, 254],
    'auto_class_weights': ['None', 'Balanced', 'SqrtBalanced'], # For Imbalance
    'leaf_estimation_method': ['Newton', 'Gradient'],
    'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS']
}

# --- C. LightGBM Classification (Deep Grid) ---
LIGHTGBM_CLS_GRID = {
    'n_estimators': [100, 200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [20, 31, 50, 100],
    'max_depth': [-1, 10, 20, 30],
    'min_child_samples': [10, 20, 30, 50],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'is_unbalance': [True, False], # For Imbalance
    'class_weight': [None, 'balanced'],
    'boosting_type': ['gbdt', 'dart']
}

# --- D. Random Forest Classification (Deep Grid) ---
RF_CLS_GRID = {
    'n_estimators': [100, 200, 300, 500, 800],
    'max_depth': [None, 10, 20, 40, 60],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', None], # Critical for Imbalance
    'criterion': ['gini', 'entropy', 'log_loss']
}

# --- E. MLP Classification (Deep Grid) ---
MLP_CLS_GRID = {
    'hidden_layer_sizes': [
        (50,), (100,), (100, 50), (100, 50, 25), 
        (200, 100), (256, 128, 64), (512, 256, 128, 64)
    ],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01],
    'batch_size': ['auto', 64, 128],
    'shuffle': [True, False],
    'tol': [1e-3, 1e-4, 1e-5]
}