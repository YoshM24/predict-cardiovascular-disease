#!/usr/bin/env python
# coding: utf-8

# Import pandas and numpy libraries along with display function from IPython
import pandas as pd
import numpy as np
from IPython.display import display

# Imports for data visualization
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

# Import train_test_split to split the data
from sklearn.model_selection import train_test_split

# Import SMOTE for balancing the dataset
from imblearn.over_sampling import SMOTE

# Import Classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Import functions to plot ROC curve
from sklearn.metrics import roc_curve, RocCurveDisplay

# Import tensorflow for deep learning
import tensorflow

# Import config file
from config_val import FILE_LOC, CAT_COLS, CONT_COLS, TARGET_COL, CV_VAL, CV_N_SPLITS, RANDOM_STATE_VAL, SMOTE_RANDOM_STATE_VAL, N_JOBS_VAL, N_ITER_VAL, TRAIN_TEST_SPLIT_SIZE, SNS_FIG_HEIGHT, SNS_FIG_WIDTH, TENYEARCHD_DEF, SEX_DEF, EDUCATION_DEF, CURRENTSMOKER_DEF, BPMEDS_DEF, PREVALENTSTROKE_DEF, PREVALENTHYP_DEF, DIABETES_DEF, KF_CROSS, CLASSIFIER_LIST_DICT, RF_PARAM, KN_PARAM, GBT_PARAM

# Import file containing functions
from function_util import draw_pie_chart, draw_plotly_histogram, draw_plotly_single_feature_boxplot, draw_plotly_multi_feature_boxplot, draw_seaborn_histogram, draw_seaborn_barplot, draw_seaborn_heatmap, data_scaling, run_LassoCV, outlier_handle_iqr, run_classifier, draw_ROC_Curve, model_comparison_by_cross_val, run_hyperparameter_tuning, gen_summary_table, run_hyperparameter_tuning_dl, run_dl_model

# Configure seaborn diagram size as per values given in configuration file
sns.set(rc={'figure.figsize':(SNS_FIG_HEIGHT, SNS_FIG_WIDTH)})


# Read the heart disease data file
chd_df = pd.read_csv(FILE_LOC)
display(chd_df)


## Rename male column as sex to avoid confusion

# Initially, take a copy of the original dataset to avoid changes to the original
chd_df_copy = chd_df.copy()

chd_df_copy.rename(columns = {'male':'sex'}, inplace = True)
display(chd_df_copy)


# Retrieve information about the dataset
print(chd_df_copy.info(), end="\n\n")


# BPMeds is a categorical variable. Hence, type of the variable to be converted to integer instead of float
chd_df_copy = chd_df_copy.astype({'BPMeds':'int'})
print(chd_df_copy.info(), end="\n\n")


# Separate the categorical, continuous and target columns
cat_cols = CAT_COLS
print(f"Categorical columns (count={len(cat_cols)}): {cat_cols}")

cont_cols = CONT_COLS
print(f"Continuous columns (count={len(cont_cols)}): {cont_cols}")

target_col = TARGET_COL
print(f"Target column (count={len(target_col)}): {target_col}", end="\n\n")


# Check for missing values (based on null cells)
missing_values = chd_df_copy.isnull().sum()
print(f"Display missing values under each column: \n{missing_values}", end="\n\n")


# Check for values having special characters (outside the types specified for each column). Here, we will check for unique values in all columns to detect if there are irrelavant values.
for col in chd_df_copy:
    print(f"{col}: {chd_df_copy[col].unique()}", end="\n\n")


# Generate statistical summary of dataset
print("Statistical Summary of Dataset")
display(chd_df_copy.describe())
print()


# Check for row duplicates
unique_rows_in_df = chd_df_copy.drop_duplicates()
print(f"Total number of duplicate rows = {len(chd_df_copy) - len(unique_rows_in_df)}", end="\n\n")
# Since there are no duplicate rows, we will NOT be performing duplicate row operations


# Define categorical variables
TenYearCHD_def = TENYEARCHD_DEF
sex_def = SEX_DEF
education_def = EDUCATION_DEF
currentSmoker_def = CURRENTSMOKER_DEF
BPMeds_def = BPMEDS_DEF
prevalentStroke_def = PREVALENTSTROKE_DEF
prevalentHyp_def = PREVALENTHYP_DEF
diabetes_def = DIABETES_DEF


# Composition of target values
target_count = chd_df_copy['TenYearCHD'].value_counts()
draw_pie_chart(target_count, target_count.index.map(TenYearCHD_def), '%1.1f%%')

## Above visualizations suggest that there is a large unbalance between the two target values.
## Hence, Oversampling or undersampling to be considered when training the dataset.


'''
Value distribution by age and sex
'''

# Create pairplots between different parameters in the dataset
_ = sns.pairplot(chd_df_copy)


# Histogram for distribution of values for sex
draw_plotly_histogram(data=chd_df_copy, x="sex", x_title="Sex",  y_title='Count', title="Distribution of Sex",
                      template="simple_white", xaxis_dict=sex_def, auto_text=True)


# Histogram for age distribution in entire dataset
draw_seaborn_histogram(data=chd_df_copy, x="age", title="Distribution of Age", color="#7490c0",
                       bins=len(chd_df_copy['age'].unique()), show_y_values=True)


# Histogram for age distribution based on both sex.
draw_seaborn_histogram(data=chd_df_copy, x="age", hue="sex", title="Distribution of Age by Sex",
                       bins=len(chd_df_copy['age'].unique()), show_y_values=True, multiple="dodge",
                       legend_title="Sex", change_legend_text=True, legend_label_dict=sex_def)


# Histogram for age distribution based on target column.
draw_seaborn_histogram(data=chd_df_copy, x="age", hue="TenYearCHD", title="Distribution of Age by TenYearCHD",
                       bins=len(chd_df_copy['age'].unique()), show_y_values=True, multiple="dodge",
                       change_legend_text=True, legend_label_dict=TenYearCHD_def)


# Boxplot for TenYearCHD target column based on age and sex
draw_plotly_single_feature_boxplot(data=chd_df_copy, x="TenYearCHD", y="age",
                                   title="Boxplot for TenYearCHD based on Age and Sex",
                                   x_title='TenYearCHD', y_title='Age', color='sex',
                                   xaxis_dict=TenYearCHD_def, legend_title="Sex",
                                   legend_label_dict=sex_def, change_legend_text=True)


# Histogram for age distribution based on education.
draw_seaborn_histogram(data=chd_df_copy, x="age", hue="education", title="Distribution of Age by Education",
                       bins=len(chd_df_copy['age'].unique()), legend_title="Education", show_y_values=True,
                       multiple="dodge", change_legend_text=True, legend_label_dict=education_def)


# Histogram for distribution of education with target column
draw_plotly_histogram(data=chd_df_copy, x="education", x_title='Education', y_title='Count',
                      title="Distribution of Education by TenYearCHD", color='TenYearCHD',
                      template="simple_white", barmode='group', xaxis_dict=education_def,
                      legend_title="TenYearCHD", legend_label_dict=TenYearCHD_def, change_legend_text=True, auto_text=True)

'''
 ./Value distribution by age and sex
'''

'''
EDA of Continuous Variables
'''

# Mean values grouped by age and target
mean_age_target_df = chd_df_copy.groupby(['age', 'TenYearCHD']).mean().reset_index()
display(mean_age_target_df)


# Distribution of total cholesterol with age
draw_seaborn_barplot(data=mean_age_target_df, x="age", y='totChol', hue='TenYearCHD', x_label="Age",
                     y_label="Average Total Cholesterol",
                     title="Average Total Cholesterol Level for Each Age by TenYearCHD",
                     legend_title="TenYearCHD", change_legend_text=True, legend_label_dict=TenYearCHD_def)


# Distribution of glucose with age
draw_seaborn_barplot(data=mean_age_target_df, x="age", y='glucose', hue='TenYearCHD', x_label="Age",
                     y_label="Average Glucose", title="Average Glucose Level for Each Age by TenYearCHD",
                     legend_title="TenYearCHD", change_legend_text=True, legend_label_dict=TenYearCHD_def)

fig14, axes = plt.subplots(3, 3, figsize=(20, 10), sharey=False)

# axes
sns.histplot(ax=axes[0,0], data=chd_df_copy, x="cigsPerDay")
axes[0,0].set_title('Histogram of CigsPerDay')

sns.histplot(ax=axes[0,1], data=chd_df_copy, x="totChol")
axes[0,1].set_title('Histogram of Total Cholesterol level')

sns.histplot(ax=axes[0,2], data=chd_df_copy, x="sysBP")
axes[0,2].set_title('Histogram of Systolic BP')

sns.histplot(ax=axes[1,0], data=chd_df_copy, x="diaBP")
axes[1,0].set_title('Histogram of Diastolic BP')

sns.histplot(ax=axes[1,1], data=chd_df_copy, x="BMI")
axes[1,1].set_title('Histogram of BMI')

sns.histplot(ax=axes[1,2], data=chd_df_copy, x="heartRate")
axes[1,2].set_title('Histogram of HeartRate')

sns.histplot(ax=axes[2,0], data=chd_df_copy, x="glucose")
axes[2,0].set_title('Histogram of Glucose')

axes[2,2].set_axis_off()
axes[2,1].set_axis_off()

fig14.tight_layout()
fig14.show()

'''
 ./ EDA of Continuous Variables
'''


'''
EDA for discrete variables (EXCEPT age and education)
'''

for i in cat_cols:    
    if(i not in ['sex', 'education']):
        draw_plotly_histogram(data=chd_df_copy, x=i, y_title='Count', color='TenYearCHD', template="simple_white",
                              barmode='group', title="Distribution of "+str(i)+" with TenYearCHD",
                              xaxis_dict=locals()[str(i)+'_def'], legend_title="TenYearCHD",
                              legend_label_dict=TenYearCHD_def, change_legend_text=True, auto_text=True)

# Obtain the correlation matrix for all values in the dataset
ax = draw_seaborn_heatmap(chd_df_copy)

## From the above correlation matrix, it is understood that non of the parameters show a strong correlation (>=0.9) against one another.

'''
 ./ EDA for discrete variables (EXCEPT age and education)
'''


'''
Feature Selection (Pre-Standardization) using Lasso Coefficient
'''

lasso_result = run_LassoCV(chd_df_copy.drop(['TenYearCHD'], axis=1), chd_df_copy[['TenYearCHD']])

selected_features = lasso_result['selected_features']
eliminated_features = lasso_result['eliminated_features']

print(f"Selected features: {selected_features}")
print(f"Eliminated features: {eliminated_features}", end="\n\n")


# Update dataset by removing the features having a Lasso coefficient of zero
chd_df_backup = chd_df_copy.copy()
chd_df_copy = chd_df_copy.drop(eliminated_features, axis=1)
print(chd_df_copy, end="\n\n")

'''
 ./ Feature Selection (Pre-Standardization) using Lasso Coefficient
'''


'''
Separating categorical, continuous and target columns for analysis
'''

# Separate the categorical, continuous and target columns
print(f"Original categorical columns: {cat_cols}")
cat_cols = list(set(cat_cols) - set(eliminated_features))
print(f"Updated categorical columns: {cat_cols}", end="\n\n")

print(f"Original continuous columns: {cont_cols}")
cont_cols = list(set(cont_cols) - set(eliminated_features))
print(f"Updated continuous columns: {cont_cols}", end="\n\n")

print(f"Target column: {target_col}", end="\n\n")

'''
 ./ Separating categorical, continuous and target columns for analysis
'''


'''
Split data for training and testing
'''

# Perform Trains Test Split
X = chd_df_copy.drop(TARGET_COL, axis=1)
y = chd_df_copy[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=TRAIN_TEST_SPLIT_SIZE,stratify=y,
                                                    random_state=RANDOM_STATE_VAL)

'''
 ./ Split data for training and testing
'''


'''
Handling outliers
'''

# Detect outliers in all continuous features
draw_plotly_multi_feature_boxplot(X_train, cont_cols, title="Boxplot for all Continuous Features")


## Removing outliers

# Combine X_train and y_train to form complete training dataframe
chd_df_new = X_train.copy()
chd_df_new = chd_df_new.assign(TenYearCHD = y_train.copy().values)

for col in sorted(cont_cols):   
    if col != 'age':
        chd_df_new = outlier_handle_iqr(chd_df_new, chd_df_new[col])


# Create boxplots for updated dataset to check for outliers in all continuous features
draw_plotly_multi_feature_boxplot(chd_df_new, cont_cols, title="Boxplot for all Continuous Features (Post Outlier Removal)")


# Update the training values
X_train = chd_df_new.drop('TenYearCHD', axis=1)
y_train = pd.DataFrame(chd_df_new['TenYearCHD'], columns = ['TenYearCHD'])

'''
 ./ Handling outliers
'''


'''
Standardize the dataset
'''

# Separate the target column and feature columns in updated training dataset. Afterwards, scale the training and testing data.
trainX, testX = data_scaling(X_train, X_test, 'standard')

X_train = pd.DataFrame(trainX, columns = chd_df_new.drop('TenYearCHD', axis=1).columns)
X_test = pd.DataFrame(testX, columns = chd_df_new.drop('TenYearCHD', axis=1).columns)

print(X_train, end="\n\n")
print(X_test, end="\n\n")

'''
 ./ Standardize the dataset
'''


'''
Balancing the dataset
'''

# Balancing dataset using SMOTE
sm = SMOTE(random_state = SMOTE_RANDOM_STATE_VAL)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.values.ravel())

print(f"Shape of X_train before SMOTE = {X_train.shape}")
print(f"Shape of y_train before SMOTE = {y_train.shape}")

print(f"Shape of X_train after SMOTE = {X_train_res.shape}")
print(f"Shape of y_train after SMOTE = {y_train_res.shape}", end="\n\n")

print("Before OverSampling, counts of label '1': {}".format(sum(y_train.values.ravel() == 1)))
print("Before OverSampling, counts of label '0': {}".format(sum(y_train.values.ravel() == 0)))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

'''
 ./ Balancing the dataset
'''

'''
Run classifiers under default hyperparameters
'''

## Perform classification for all classifiers under dfeault hyperparamters and generate their results along with ROC curve
classifier_list_dict = CLASSIFIER_LIST_DICT

classifier_results={}
count = 1
for key,model in classifier_list_dict.items():
    print(f"Results for {key}", end="\n\n")
    classifier_results[key] = run_classifier(model, X_train_res, y_train_res.ravel(), X_test, y_test)
    draw_ROC_Curve(model, count, X_test, y_test, len(classifier_list_dict))
    count = count+1 


# Generate summary table for classifer results
gen_summary_table(classifier_results)


## Comparison of ML Models under default hyperparameters
comp_models = model_comparison_by_cross_val(cls_dict=classifier_list_dict, X=X_train_res, Y=y_train_res, scoring='accuracy', title='ML Model Comparison using Results for Accuracy under Stratified Cross Validation')

'''
 ./ Run classifiers under default hyperparameters
'''


'''
Perform hyperparameter tuning (Optimization) for each classifier
'''

## Hyperparameter tuning for Random Forest Classifier

# Obtain the best parameters and re-run classifier
rf_param = RF_PARAM
rf_gsearch_result = run_hyperparameter_tuning(searchCV='gridSearch', classifier=RandomForestClassifier(),
                                              train_X=X_train_res, train_Y=y_train_res.ravel(), params=rf_param)
print(f"Best hyperparameters for random forest = {rf_gsearch_result.best_params_}", end="\n\n")

rf_post_tuning = RandomForestClassifier(**rf_gsearch_result.best_params_, random_state=RANDOM_STATE_VAL)


## Hyperparameter tuning for KNeighbours Classifier

# Obtain the best parameters and re-run classifier
kn_param = KN_PARAM
kn_gsearch_result = run_hyperparameter_tuning(searchCV='gridSearch', classifier=KNeighborsClassifier(),
                                              train_X=X_train_res, train_Y=y_train_res.ravel(), params=kn_param)
print(f"Best hyperparameters for kneighbours = {kn_gsearch_result.best_params_}", end="\n\n")

kn_post_tuning = KNeighborsClassifier(**kn_gsearch_result.best_params_)


## Hyperparameter tuning for Gradient Boosting Trees Classifier

# Obtain the best parameters and re-run classifier
gbt_param = GBT_PARAM
gbt_gsearch_result = run_hyperparameter_tuning(searchCV='gridSearch',
                                               classifier=GradientBoostingClassifier(random_state=RANDOM_STATE_VAL),
                                               train_X=X_train_res, train_Y=y_train_res.ravel(), params=gbt_param)
print(f"Best hyperparameters for gradient boosting trees = {gbt_gsearch_result.best_params_}", end="\n\n")

gbt_post_tuning = GradientBoostingClassifier(**gbt_gsearch_result.best_params_, random_state=RANDOM_STATE_VAL)

'''
 ./ Perform hyperparameter tuning (Optimization) for each classifier 
'''


'''
Run classifiers with their best parameters and view their performance
'''

tuned_classifier_list_dict = {
    'Random Forest': rf_post_tuning,
    'K-Neighbours': kn_post_tuning,
    'Gradient Boosting Trees': gbt_post_tuning
    }

tuned_classifier_results={}
count=1
for key,val in tuned_classifier_list_dict.items():
    print(f"Results for {key} after tuning", end="\n\n")
    tuned_classifier_results[key] = run_classifier(val, X_train_res, y_train_res.ravel(), X_test, y_test)
    draw_ROC_Curve(val, count, X_test, y_test, len(tuned_classifier_list_dict))
    count = count+1

'''
 ./ Run classifiers with their best parameters and view their performance
'''


'''
Summarize the performance of optimized classifiers
'''

gen_summary_table(tuned_classifier_results)

'''
 ./ Summarize the performance of optimized classifiers
'''


'''
Develop and run deep learning model
'''

## Run deep learning model
input_dimensions = X_train_res.shape[1]

dl_output = run_dl_model(n_layers=3, first_layer_nodes=input_dimensions, hidden_layer_nodes=6,
                         input_dimensions=input_dimensions, drop1=0.2, drop2=0.5,
                         loss_func='binary_crossentropy', X_train=X_train_res, Y_train=y_train_res,
                         X_test=X_test, Y_test=y_test, epochs=100, batch_size=32)


# Plot ROC curve for deep learning model
_ = RocCurveDisplay.from_predictions(y_test, dl_output['predicted_values'])

'''
 ./ Develop and run deep learning model
'''


'''
Optimize the deep learning model and generate results
'''

## Tune Deep learning function
input_dimensions = X_train_res.shape[1]
hidden_layer_nodes = np.unique(np.linspace(2, input_dimensions, 5).astype(int))

dl_best_params = run_hyperparameter_tuning_dl(layers=[3], first_layer_nodes=[input_dimensions],
                                              hidden_layer_nodes=hidden_layer_nodes,
                                              input_dimensions=[input_dimensions], X=X_train_res, Y=y_train_res)
print(dl_best_params)


## Run the deep learning model with the best parameters
tensorflow.keras.utils.set_random_seed(RANDOM_STATE_VAL)
tensorflow.config.experimental.enable_op_determinism()
dl_output_tuned = run_dl_model(**dl_best_params, X_train=X_train_res, Y_train=y_train_res, X_test=X_test, Y_test=y_test)


# Plot ROC curve for optimized deep learning model
_ = RocCurveDisplay.from_predictions(y_test, dl_output_tuned['predicted_values'])

'''
 ./ Optimize the deep learning model and generate results
'''