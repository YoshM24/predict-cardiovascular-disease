# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 22:27:31 2023

@author: SINGER
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score

# Specify folder path
FILE_LOC = "data/CHD_data.csv"              # Specify the path of the dataset


# Define columns in datatet
DATA_COL_DEFS = {
    'male': "Sex of the person; (Binary); 0: Female, 1: Male",
    'age': "Age of the patient; (Continuous)",
    'education': "0: Less than High School and High School degrees, 1: College Degree and Higher",
    'currentSmoker': "Whether or not a person is a current smoker; (Binary); 0: No, 1: Yes",
    'cigsPerDay': "The number of cigarettes that the person smoked on average in one day; (Continuous)",
    'BPMeds': "Whether or not the patient was on blood pressure medication; (Binary); 0: No, 1: Yes",
    'prevalentStroke': "Whether or not the patient had previously had a stroke; (Binary); 0: No, 1: Yes",
    'prevalentHyp': "Whether or not the patient was hypertensive; (Binary); 0: No, 1: Yes",
    'diabetes': "Whether or not the patient had diabetes; (Binary); 0: No, 1: Yes",
    'totChol': "Total cholesterol level; (Continuous)",
    'sysBP': "Systolic blood pressure; (Continuous)",
    'diaBP': "Diastolic blood pressure; (Continuous)",
    'BMI': "Body Mass Index; (Continuous)",
    'heartRate': "Heart rate; (Continuous)",
    'glucose': "Glucose level; (Continuous)",
    'TenYearCHD': "Target Column - 10 year risk of coronary heart disease; (Binary); 0: Absent, 1: Present"
}


# Define the categorical, continuous and target columns of the dataset
CAT_COLS = ['sex', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']   # Categorical columns
CONT_COLS = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']               # Continuous columns
TARGET_COL = ['TenYearCHD']                                                                                 # Target column 
    

# Define values denoted by categorical columns
TENYEARCHD_DEF = {0: 'Negative', 1: 'Positive'}
SEX_DEF = {0:'Female', 1:'Male'}
EDUCATION_DEF = {0:'Upto High School Degree', 1:'College Degree and Higher'}
CURRENTSMOKER_DEF = {0:'No', 1:'Yes'}
BPMEDS_DEF = {0:'No', 1:'Yes'}
PREVALENTSTROKE_DEF = {0:'No', 1:'Yes'}
PREVALENTHYP_DEF = {0:'No', 1:'Yes'}
DIABETES_DEF = {0:'No', 1:'Yes'}


# General constants
CV_VAL = 10
CV_N_SPLITS = 10
RANDOM_STATE_VAL = 42
SMOTE_RANDOM_STATE_VAL = 2
N_ITER_VAL = 1000
TRAIN_TEST_SPLIT_SIZE = 0.2


# Dimensions for seaborn plots
SNS_FIG_HEIGHT = 14.7
SNS_FIG_WIDTH = 8.27


# Parameters for heatmap
HEATMAP_VMAX = 1.0
HEATMAP_VMIN = -1.0
HEATMAP_ANNOT = True


# Default conditions for stratified kfold cross-validation
KF_CROSS = StratifiedKFold(shuffle=True, n_splits=CV_N_SPLITS, random_state=RANDOM_STATE_VAL)


# Initialize classifiers
RF = RandomForestClassifier(random_state=RANDOM_STATE_VAL)          # Initialize random forest classifier
KN = KNeighborsClassifier(n_neighbors=3)                            # Initialize random kneighbours classifier
GBT = GradientBoostingClassifier(random_state=RANDOM_STATE_VAL)     # Initialize random gradient boosting trees classifier


# Dictionary of classifiers
CLASSIFIER_LIST_DICT = {
    'Random Forest': RF,
    'K-Neighbours': KN,
    'Gradient Boosting Trees': GBT
    }


# Hyperparameters for tuning Random Forest
RF_PARAM = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [None, 2, 3, 5, 10, 15, 20],
    "min_samples_leaf": [5, 10, 15, 20, 50],
    "max_features": ['auto', 'sqrt', 'log2', None],
    "criterion": ['gini', 'entropy']
    }    


# Hyperparameters for tuning KNeighbours
KN_PARAM = {
    "leaf_size": list(range(1,20)),
    "n_neighbors": list(range(1,20)),
    "p": [1,2]
    }


# Hyperparameters for tuning Gradient Boosting Trees
GBT_PARAM = {
    "n_estimators": [10, 50, 100],
    "max_depth": [3, 5, 8],
    "min_samples_split": np.linspace(0.1, 0.5, 5),
    "min_samples_leaf": np.linspace(0.1, 0.5, 5),
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1],
    "max_features":["log2", "sqrt"]
    }


# GridsearchCV parameters
SCORING_VAL = {
    "recall": make_scorer(recall_score, average = 'macro')
    }
REFIT_VAL = "recall"
N_JOBS_VAL = 3


# Deep learning model hyperparameters
DL_LOSS_FUNC_VALS = ['binary_crossentropy']
DL_COMPILE_OPTIMIZER = 'adam'
INPUT_ACIVATION = 'relu'
HIDDEN_ACIVATION = 'relu'
OUTPUT_ACIVATION = 'sigmoid'
DROP1_VAL = [0.1, 0.2]
DROP2_VAL = [0.3, 0.4, 0.5]
EPOCHS_RANGE = [50, 100]
BATCH_SIZE_RANGE = [32, 64]
