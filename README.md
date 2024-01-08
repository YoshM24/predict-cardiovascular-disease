The data for the study was obtained from Kaggle (Link: https://www.kaggle.com/datasets/captainozlem/framingham-chd-preprocessed-data)

The project was initially developed in Jupyter Notebook to observe visualizations. The code was later optimized for a Python console project and uploaded to this repository.

The project consists of 3 files:
  1. classify_chd_main.py - the main file
  2. config_val.py - contains all the configuration values
  3. function_util.py - contains the author defined functions that will be called from the main file (outside library functions)

The study aims at finding the performance of 3 machine learning (ML) classifiers and a deep learning (DL) model in predicting cardiovascualr disease.

The three ML classifiers are:
  1. RandomForestClassifier
  2. KNeighboursClassifier
  3. GradientBoostingClassifier

For the the DL model, a 3 layered perceptron is used in this example.

The code initially produces visualizations to identify distribution of the target column as well as several other parameters.

Highly correlated parameters are removed from the dataset with reference to the correlation matrix and Least Absolute Shrinkage and Selection Operator (LASSO) coefficients. Correlation matrix helps us understand the close relationship between features, whereas LASSO enables the identification of parameters that show no association with the column of interest (target column). These parameters produce a ZERO for LASSO coefficients.

In this example, outliers in the dataset are identified using boxplots and are removed providing an upper and lower limit based on the Inter-Quartile Range (IQR).

As initial visualizations displayed that the dataset was highly imbalanced for the target values, the code also has a section for balancing the dataset using SMOTE (Oversampling).

The data is split for training and testing prior to passing into the classifiers. In this example, 80% of the data was used for training as more values for training would capture more variations by the model.

Hyperparameter tuning and changes in prediction of the classifiers after tuning too are covered in this example for all ML and DL models.
