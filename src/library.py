# --- Basic Data Handling & Visualization ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import time
import joblib

# --- GUI (For file selection) ---
from tkinter import filedialog
import tkinter as tk

# --- Machine Learning: Regression (For Binding Affinity) ---
# Used for predicting in-silico binding affinity
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# --- Machine Learning: Classification (For In-Vitro Inhibition) ---
# Used for predicting active/inactive compounds
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# --- Data Preprocessing & Model Selection ---
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.utils import resample

# --- Metrics (Evaluation) ---
from math import sqrt
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --- Statistics ---
from scipy.stats import norm
from scipy import stats

# --- Warnings ---
import warnings
warnings.filterwarnings('ignore')

# --- Chemistry Tools (RDKit & Meeko) ---
# Required for ligand preparation as detailed in Supplementary Info
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation
except ImportError:
    print("Warning: RDKit or Meeko not found. Ligand preparation will not work.")

print("All libraries loaded successfully from src.library")