# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 21:29:32 2023

@author: SINGER
"""
import matplotlib
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.io as pio
from plotly.offline import plot
pio.renderers.default = 'browser'               # Display plotly plots in default browser

import tensorflow

from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from config_val import CV_VAL, CV_N_SPLITS, RANDOM_STATE_VAL, SMOTE_RANDOM_STATE_VAL, N_ITER_VAL, TRAIN_TEST_SPLIT_SIZE, KF_CROSS, HEATMAP_VMAX, HEATMAP_VMIN, HEATMAP_ANNOT, SCORING_VAL, REFIT_VAL, N_JOBS_VAL, DL_COMPILE_OPTIMIZER, INPUT_ACIVATION, HIDDEN_ACIVATION, OUTPUT_ACIVATION, DL_LOSS_FUNC_VALS, DROP1_VAL, DROP2_VAL, EPOCHS_RANGE, BATCH_SIZE_RANGE


# Draw a piechart using matplotlib
def draw_pie_chart(data, labels, prct):
    plt.pie(data, labels=labels, startangle=90, autopct=prct)
    plt.tight_layout()
    plt.show()


# Draw a histogram using plotly
def draw_plotly_histogram(data, x, y_title=None, x_title=None, title=None, color=None, template=None, color_seq=None, barmode='relative', xaxis_dict={}, legend_title=None, legend_label_dict={}, change_legend_text=False, auto_text=False):    
    fig = px.histogram(data, x=x, title=title, color=color, template=template, barmode=barmode, color_discrete_sequence=color_seq, text_auto=auto_text)
    
    if(xaxis_dict):
        fig.update_layout(xaxis = dict(
                      tickmode = 'array',
                      tickvals = list(xaxis_dict.keys()),
                      ticktext = list(xaxis_dict.values())
                  ))
        
    fig.update_layout(legend=dict(
                      bordercolor="Gray",
                      borderwidth=2
                  ))
    if(y_title):
        fig.update_layout(yaxis_title=y_title)
    if(x_title):
        fig.update_layout(xaxis_title=x_title)
    if(legend_title):
        fig.update_layout(legend = dict(
                      title=legend_title
                  ))
    
    if(change_legend_text):
        for indx in fig.data:
            indx.hovertemplate = 'TenYearCHD='+legend_label_dict[int(indx.name)]+'<br>'+fig.layout.xaxis.title.text+'=%{x}<br>'+fig.layout.yaxis.title.text+'=%{y}<extra></extra>'
            indx.name = legend_label_dict[int(indx.name)]

    fig.show()


# Draw a boxplot using plotly for a single feature as x
def draw_plotly_single_feature_boxplot(data, x, y, y_title=None, x_title=None, title=None, color=None, template=None, color_seq=None, xaxis_dict={}, legend_title=None, legend_label_dict={}, change_legend_text=False):
    fig = px.box(data, x=x, y=y, title=title, color=color, template=template, color_discrete_sequence=color_seq)
    
    if(xaxis_dict):
        fig.update_layout(xaxis = dict(
                      tickmode = 'array',
                      tickvals = list(xaxis_dict.keys()),
                      ticktext = list(xaxis_dict.values())
                  ))
        
    fig.update_layout(legend=dict(
                      bordercolor="Gray",
                      borderwidth=2
                  ))
    if(y_title):
        fig.update_layout(yaxis_title=y_title)
    if(x_title):
        fig.update_layout(xaxis_title=x_title)
    if(legend_title):
        fig.update_layout(legend = dict(title=legend_title))
    
    if(change_legend_text):
        for indx in fig.data:
            indx.name = legend_label_dict[int(indx.name)]

    fig.show()


# Draw a boxplot using plotly for multiple features as x
def draw_plotly_multi_feature_boxplot(df, cols):
    fig = go.Figure()

    for col in cols:
        fig.add_trace(go.Box(y=df[col].values, name=df[col].name))

    fig.show()
    

# Draw a histogram using seaborn
def draw_seaborn_histogram(data, x, y=None, hue=None, title=None, color="#7490c0", bins='auto', show_y_values=False, multiple="layer", legend_title=None, change_legend_text=False, legend_label_dict={}, set_x = False):
    ax = sns.histplot(data=data, x=x, y=y, hue=hue, color=color, bins=bins, multiple=multiple)
    if(set_x):
        ax.set_xticks(x)

    if(show_y_values):
        for p in ax.patches:
            t = ax.annotate(str(p.get_height()), xy = (p.get_x(), p.get_height() + 1))
            t.set()
    
    if(legend_title):
        ax.get_legend().set_title(legend_title)
       
    if(title):
        ax.set_title(title)
    
    if(change_legend_text):
        try:
            legend_text = ax.axes.get_legend().texts
            for i in legend_text:
                i.set_text(legend_label_dict[int(i.get_text())])
                
        except Exception as e: 
            print(e)

            
# Draw a barplot using seaborn
def draw_seaborn_barplot(data, x, y=None, hue=None, show_y_values=False, x_label=None, y_label=None, title=None, legend_title=None, change_legend_text=False, legend_label_dict={}):
    ax = sns.barplot(data=data, x=x, y=y, hue=hue)
    if(title):
        plt.title(title)
    if(x_label):
        plt.xlabel(x_label)
    if(y_label):
        plt.ylabel(y_label)
    if(show_y_values):
        for p in ax.patches:
            t = ax.annotate(str(p.get_height()), xy = (p.get_x(), p.get_height() + 1))
            t.set()
            
    if(legend_title):
        ax.get_legend().set_title(legend_title)
        
    if(change_legend_text):
        try:
            legend_text = ax.axes.get_legend().texts
            for i in legend_text:
                i.set_text(legend_label_dict[int(i.get_text())])
                
        except Exception as e: 
            print(e)
    plt.show()


# Draw a heatmap using seaborn
def draw_seaborn_heatmap(data):
    ax = sns.heatmap(data.corr(), vmax=HEATMAP_VMAX, vmin=HEATMAP_VMIN, annot = HEATMAP_ANNOT)
    return ax


# Standardize the data
def data_scaling(train_X, test_X, scaler='standard'):
    if(scaler=='standard'):
        # Scale using StandardScaler
        scaling = StandardScaler()
    elif(scaler=='minmax'):
        # Scale using MinMaxScaler
        scaling = MinMaxScaler()
    else:
        raise ValueError("Invalid scaler. Please specify if you wish to carry out standard or minmax scaler.")
    
    try:
        X_train = scaling.fit_transform(train_X)
        X_test = scaling.transform(test_X)
    except Exception as e: 
        print(e)
    
    return X_train, X_test


# Execute LassoCV for Feature Selection
def run_LassoCV(X, Y):
    res = {}
    try:
        reg = LassoCV(cv=KF_CROSS, n_jobs=N_JOBS_VAL)

        reg.fit(X, Y.values.ravel())
        print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
        print("Best score using built-in LassoCV: %f" %reg.score(X, Y), end="\n\n")

        # Obtain the coefficients and output the vatiables with coeffcients equal to and not equal to zero.
        coef = pd.Series(reg.coef_, index = X.columns)
        print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables", end="\n\n")

        # Sort and get the LASSO coefficents
        imp_coef = coef.sort_values()

        imp_coef_df = pd.DataFrame({'col_features':imp_coef.index, 'vals':imp_coef.values})

        selected_features = list(imp_coef_df.loc[imp_coef_df['vals'] != 0]['col_features'])
        eliminated_features = list(imp_coef_df.loc[imp_coef_df['vals'] == 0]['col_features'])

        # Plot a horizontal bar graph displaying the obtained LASSO coefficients
        fig = px.bar(imp_coef_df, x='vals', y='col_features', orientation='h', title='Lasso Coefficients for Features in Dataset')

        fig.update_layout(yaxis_title="Features", xaxis_title="Coefficient")
        fig.show()

        res = {
            "feature_coefficients": imp_coef_df,
            "selected_features": selected_features,
            "eliminated_features": eliminated_features
        }
    except Exception as e: 
        print(e)
    
    return res


# Handling outliers based on inter-quartile range
def outlier_handle_iqr(complete_df, continous_outlier_column_df):
    df = complete_df

    # Compute the first and third quartiles for dataframe with continuous data column having outliers
    cont_col_df = continous_outlier_column_df

    q1 = cont_col_df.quantile(0.25)
    q3 = cont_col_df.quantile(0.75)

    # Calculate of interquartile range
    IQR = q3 - q1

    # Specify upper and lower limit
    low_lim = q1 - 1.5 * IQR
    upp_lim = q3 + 1.5 * IQR

    try:
        # Update dataframe 'df' by removing the rows where the values of the specified column are beyond the upper and lower limits
        index = df[(cont_col_df < low_lim) | (cont_col_df > upp_lim)].index
        df.drop(index, inplace=True)
    except Exception as e: 
        print(e)
            
    return df


# Execute the classifier passed to the function as 'clr'
def run_classifier(clr, train_X, train_Y, test_X, test_Y):
    clr.fit(train_X, train_Y)
    
    # Make predictions for both training and testing data
    pred_Y_train = clr.predict(train_X)
    pred_Y_test =  clr.predict(test_X)
    
    # Get accuracies for both training and testing data
    model_train_acc = accuracy_score(train_Y, pred_Y_train) 
    model_test_acc = accuracy_score(test_Y, pred_Y_test)
    
    # Get precision, recall and f1 score values for test data
    model_test_precision = precision_score(test_Y, pred_Y_test)
    model_test_recall = recall_score(test_Y, pred_Y_test)
    model_test_f1 = f1_score(test_Y, pred_Y_test)
    
    # Get classification report
    class_report = classification_report(test_Y, pred_Y_test)
    
    # Get confusion matrix for both training and testing data
    conf_matrix_train = confusion_matrix(test_Y, pred_Y_test)
    conf_matrix_test = confusion_matrix(train_Y, pred_Y_train)
    
    res = {
        "train_accuracy": model_train_acc,
        "test_accuracy": model_test_acc,
        "test_precision": model_test_precision,
        "test_recall": model_test_recall,
        "test_f1_score": model_test_f1,
        "class_report": class_report,
        "train_conf_matrix": conf_matrix_train,
        "test_conf_matrix": conf_matrix_test
    }
    
    print(f"Train accuracy: {res['train_accuracy']}")
    print(f"Test accuracy: {res['test_accuracy']}", end="\n\n")
    print(f"Test precision: {res['test_precision']}")
    print(f"Test recall: {res['test_recall']}")
    print(f"Test f1 score: {res['test_f1_score']}", end="\n\n")
    print(f"Classification report:\n {res['class_report']}", end="\n\n")
    print(f"Confusion matrix (train):\n {res['train_conf_matrix']}", end="\n\n")
    print(f"Confusion matrix (test):\n {res['test_conf_matrix']}", end="\n\n")
    print("###########################################################################", end="\n\n")
    
    return res


# Draw an ROC curve for all classifiers
def draw_ROC_Curve(classifier, count, test_X, test_Y, classifier_dict_len):
    if(count > 1):
        ax = plt.gca()
        clr_disp = RocCurveDisplay.from_estimator(classifier, test_X, test_Y, ax=ax, alpha=0.8)
    else:
        clr_disp = RocCurveDisplay.from_estimator(classifier, test_X, test_Y)
    if(count == classifier_dict_len):
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for Different Classifiers")
        plt.show()
        print("###########################################################################", end="\n\n")
        
    return count


# Compare ML Models
def model_comparison_by_cross_val(cls_dict, X, Y, scoring):
    model_results = []
    model_names = []
    
    res = {}
    
    for key, model in cls_dict.items():
        cv_results = cross_val_score(model, X, Y, cv=KF_CROSS, scoring=scoring)
        model_results.append(cv_results)
        model_names.append(key)
        msg = "%s: %f (%f)" % (key, cv_results.mean(), cv_results.std())
        print(msg, end="\n\n")
        res[key] = cv_results

    # Boxplot to compare algorithms
    fig = plt.figure()
    fig.suptitle('ML Model Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(model_results)
    ax.set_xticklabels(model_names)
    plt.show()
    
    return res


# Hyperparameter tuning based on GridSearchCV or RandomizedSearchCV
def run_hyperparameter_tuning(searchCV, classifier, train_X, train_Y, params={}):
    scoring = SCORING_VAL
    
    try:
        if(searchCV == 'gridSearch'):
            result = GridSearchCV(classifier, cv = KF_CROSS, param_grid = params, scoring=scoring, refit=REFIT_VAL, n_jobs = N_JOBS_VAL)
            result.fit(train_X, train_Y)
        elif(searchCV == 'randomSearch'):
            result = RandomizedSearchCV(classifier, param_distributions = params, n_iter = N_ITER_VAL, cv = KF_CROSS, verbose=2, random_state=RANDOM_STATE_VAL, n_jobs = N_JOBS_VAL)
            result.fit(train_X, train_Y)
        else:
            raise ValueError("Invalid searchCV. Please specify if you wish to carry out gridSearch or randomSearch for hyperparameter tuning.")
            return -1
    except Exception as e: 
        print(e)
    
    return result


# Generate summary table for classifiers
def gen_summary_table(classifier_results={}):
    classifier_summary_df = pd.DataFrame(columns=('Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1-score'))
    try:
        for key,val in classifier_results.items():
            classifier_summary_df.loc[key]=[val['train_accuracy'], val['test_accuracy'], val['test_precision'], val['test_recall'], val['test_f1_score']]
        print(classifier_summary_df)
        print("###########################################################################", end="\n\n")
    except Exception as e: 
        print(e)
            

# Create and execute a deep learning model
def run_dl_model(n_layers, first_layer_nodes, hidden_layer_nodes, input_dimensions, drop1, drop2, loss_func, X_train, Y_train, X_test, Y_test, epochs, batch_size):
    tensorflow.keras.utils.set_random_seed(RANDOM_STATE_VAL)
    tensorflow.config.experimental.enable_op_determinism()

    dl_model = Sequential()
    
    # Loop through specified number of layers
    for i in range(1, n_layers):
        if i==1:
            # Input layer
            dl_model.add(Dense(first_layer_nodes, input_dim=input_dimensions, activation=INPUT_ACIVATION))
            dl_model.add(Dropout(drop1))
        else:
            # Hidden layer/s
            dl_model.add(Dense(hidden_layer_nodes, activation=HIDDEN_ACIVATION))
            dl_model.add(Dropout(drop2))
            
    # Output layer with a single node for binary classification
    dl_model.add(Dense(1, activation=OUTPUT_ACIVATION))
    
    # Compile model
    dl_model.compile(loss='binary_crossentropy', optimizer=DL_COMPILE_OPTIMIZER, metrics=['accuracy'])

    # Fit the model
    dl_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

    # Evaluate the model
    scores = dl_model.evaluate(X_test, Y_test)
    print(scores)
    print("\n%s: %.2f%%" % (dl_model.metrics_names[1], scores[1]*100))

    # Predict and roundoff model outputs for test set
    y_res_classes = dl_model.predict(X_test, verbose=0)
    y_rounded = [round(x[0]) for x in y_res_classes]

    # Measure accuracy, precision, recall and f1 for model predictions
    accuracy = accuracy_score(Y_test, y_rounded)
    print(f"Accuracy: {accuracy}")

    precision = precision_score(Y_test, y_rounded)
    print(f"Precision: {precision}")

    recall = recall_score(Y_test, y_rounded)
    print(f"Recall: {recall}")

    f1 = f1_score(Y_test, y_rounded)
    print(f"F1_score: {f1}")
    
    res = {
        "predicted_values": y_rounded,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    return res
            

# Create deep learning model
def create_dl_model(n_layers, first_layer_nodes, hidden_layer_nodes, input_dimensions, drop1, drop2, loss_func):
    dl_model = Sequential()
    
    for i in range(1, n_layers):
        if i==1:
            # Input layer
            dl_model.add(Dense(first_layer_nodes, input_dim=input_dimensions, activation=INPUT_ACIVATION))
            dl_model.add(Dropout(drop1))
        else:
            # Hidden layer/s
            dl_model.add(Dense(hidden_layer_nodes, activation=HIDDEN_ACIVATION))
            dl_model.add(Dropout(drop2))
            
    # Output layer with a single node for binary classification
    dl_model.add(Dense(1, activation=OUTPUT_ACIVATION))
    
    try:
        # Compile keras model
        dl_model.compile(optimizer=DL_COMPILE_OPTIMIZER, loss=loss_func, metrics = ["accuracy"])
    except Exception as e: 
        print(e)
            
    return dl_model


# Hyperparameter tuning for deep learning model
def run_hyperparameter_tuning_dl(layers, first_layer_nodes, hidden_layer_nodes, input_dimensions, X, Y):
    tensorflow.keras.utils.set_random_seed(RANDOM_STATE_VAL)
    tensorflow.config.experimental.enable_op_determinism()
    
    # Wrap model into scikit-learn
    cl_model =  KerasClassifier(build_fn=create_dl_model, verbose = False) 

    # Define ranges for model loss_function, epochs and batch size
    loss_funcs = DL_LOSS_FUNC_VALS
    epochs = EPOCHS_RANGE
    batch_size = BATCH_SIZE_RANGE
    
    # Dictionary defining hyperparameters and their respective ranges to perform tuning
    dl_param_grid = dict(
        n_layers = layers,
        first_layer_nodes = first_layer_nodes,
        hidden_layer_nodes = hidden_layer_nodes,
        input_dimensions = input_dimensions,
        drop1=DROP1_VAL,
        drop2=DROP2_VAL,
        loss_func = loss_funcs,
        epochs = epochs,
        batch_size = batch_size
    )
    
    try:    
        grid = GridSearchCV(estimator = cl_model, param_grid = dl_param_grid)
        grid.fit(X,Y)
    except Exception as e: 
        print(e)
    
    print(f"Best score = {grid.best_score_}")
    print(f"Best parameters = {grid.best_params_}", end="\n\n")
    
    return grid.best_params_