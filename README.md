The data for the study was obtained from Kaggle (Link: https://www.kaggle.com/datasets/captainozlem/framingham-chd-preprocessed-data)

The study aims at finding the performance of 3 machine learning calssifiers and a deep learning model in predicting cardovascualr disease.
The three machine learning classifiers are:
  1. RandomForestClassifier
  2. KNeighboursClassifier
  3. GradientBoostingClassifier
For the the deep learning model we use a 3 layered perceptron

The code initially produces visualizations to identify distribution of target value as well as other parameters.

Highly correlated parameters are removed from the dataset. These are observed using correlation matrix and LassoCV. 

It has a section for outlier identiciation using boxplots and their removal.

The code also has a section for balancing the dataset using SMOTE

The data is split for training and testing prior to passing into classifiers

It further looks at hyperparameter tuning and changes in prediction of optimized classifiers.
