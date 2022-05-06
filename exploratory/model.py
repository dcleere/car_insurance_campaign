# script to store functions for model build
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score
import pickle
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import env

# split data in train and test, will use cross validation for validation metrics 
def data_split(df):
    
    """
    Input
    df: balanced dataset for modelling - to be split
    Output
    X_train: predictors for training
    X_val: predictors for validation
    y_train: target for training
    y_val: target for validation
    """
    
    # split the data
    X = df.drop(['CarInsurance', 'Id'], axis = 1).values

    # separate out y
    y = df['CarInsurance'].values

    # create train and val dataframes
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 42)
    
    return X_train, X_val, y_train, y_val


# create grid search function
def get_best_model(x_train, y_train, estimator, model_name, param_grid={}):
    """
    Input
    x_train: train data
    y_train: target variable
    estimator: selected estimator for grid search
    model_name: name for model output file
    param_grid: dict of hparams to be searched
    Output
    best_model: best model based on score
    best_score: best mean cross-validated score
    """
    
    # create model
    model = GridSearchCV(estimator = estimator, param_grid = param_grid, cv = 5, scoring="recall", n_jobs= 1, verbose = 1)
    
    # fit model
    model.fit(x_train, y_train)

    # get best model
    best_model = model.best_estimator_
    
    # best score
    best_score = model.best_score_ # Mean cross-validated score of the best_estimator
    
    # save best model
    pkl_filename = env.model_path+"best_{}_cv.pkl".format(model_name)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(best_model, file)
    
    print(best_model)
    
    return best_model, best_score

# create function for confusion matrix (from Kaggle)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Input
    Output
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = ['Success','Failure']

# create a model fitting function (leveraged from Kaggle)
def model_fit(x_train, y_train, x_test, y_test, cols, model, show_ft_imp = True, cv = 5):
    """
    Input
    x_train: array of input features in train data
    y_tain: array of target variable in train
    x_test: array of input features in test data]
    y_test: array of target variable in test
    cols: index of columns used in training data
    model: best estimator model object
    show_ft_imp: boolean to show feature importance plots
    cv: number of k-folds cross validation
    Output
    none: returns as plot
    """
    
    # model prediction     
    y_pred = model.predict(x_test)

    # test recall
    recall = recall_score(y_test, y_pred)
    print('Test Recall is {}'.format(recall))
    
    # model report     
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix (Test)')

    # feature importance 
    if show_ft_imp:
        feat_imp = pd.Series(model.feature_importances_, index=cols)
        feat_imp = feat_imp.nlargest(15).sort_values()
        plt.figure()
        feat_imp.plot(kind="barh",figsize=(6,8),title="Most Important Features")
