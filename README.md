The data for the study was obtained from Kaggle (Link: https://www.kaggle.com/datasets/captainozlem/framingham-chd-preprocessed-data)

The project was developed in Jupyter Notebook. The code was optimized for a Python console project and uploaded to this repository.

The project consists of 3 files:
  1. classify_chd_main.py - the main file
  2. config_val.py - contains all the configuration values
  3. function_util.py - contains the author defined functions that will be called from the main file (outside library functions)

The study aims at finding the performance of 3 machine learning classifiers and a deep learning model in predicting cardiovascualr disease.

The three machine learning classifiers are:
  1. RandomForestClassifier
  2. KNeighboursClassifier
  3. GradientBoostingClassifier

For the the deep learning model, a 3 layered perceptron is used for this example.

The code initially produces visualizations to identify distribution of the target column as well as several other parameters.

Highly correlated parameters are removed from the dataset with reference to the correlation matrix and LassoCV. 

Outliers in the dataset are identicied using boxplots and are removed.

The code also has a section for balancing the dataset using SMOTE (Oversampling).

The data is split for training and testing prior to passing into the classifiers.

It further looks at hyperparameter tuning and changes in prediction of the classifiers after tuning.
