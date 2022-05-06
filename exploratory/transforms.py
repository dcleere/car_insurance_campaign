# script to store functions for transforming the data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np

# create function to impute missing values across train and test sets    
def impute_vars(df_train):
    """
    Input
    df_train: dataframe for variables to be dropped
    Output
    df_train: dataframe with imputed vars
    """
    
    # import imputer
    cat_imp = SimpleImputer(strategy = 'most_frequent')
    
    # impute known missing
    df_train['Job'] = cat_imp.fit_transform(df_train[['Job']])
    df_train['Education'] = cat_imp.fit_transform(df_train[['Education']])
    df_train['Outcome'] = df_train['Outcome'].fillna('NoPrevContact')
    
    # check no other missing values exist
    df_null_cnt = df_train.isnull().sum(axis = 0).reset_index(name = 'null_cnt')
    cnt_more_nulls = df_null_cnt.null_cnt.sum()
    
    # raise assert if other nulls exist in the data
    assert cnt_more_nulls == 0, 'There are more missing values in your data. Please review '
    
    return df_train

# calculate call duration
def calc_call_duration(df_train_imp):
    """
    Input
    df_train_imp: dataframe with call variables
    Output
    df_train_imp2: dataframe with call duration 
    """
    
    # check if required variables exist
    a = {'CallEnd', 'CallStart'}
    b = set(df_train_imp.columns.to_list())
    if a.issubset(b):
            
        # convert to time
        df_train_imp['CallEnd'] = pd.to_datetime(df_train_imp['CallEnd'])
        df_train_imp['CallStart'] = pd.to_datetime(df_train_imp['CallStart'])
        
        # calculate duration and drop input vars
        df_train_imp['CallDuration'] = ((df_train_imp['CallEnd'] - df_train_imp['CallStart'])/np.timedelta64(1,'m')).astype(float)
        df_train_imp2 = df_train_imp.drop(['CallEnd', 'CallStart'], axis = 1)
        
    else:
        raise KeyError("Required fields are missing call duration calc")
    
    return df_train_imp2

# define encoder for education
def encode_ed(df):
    
    """
    Input
    df: dataframe with Education var for encoding
    Output
    enc_arr: array in integer format
    """
    
    # define encoder
    enc = OrdinalEncoder(categories=[['primary', 'secondary', 'tertiary']])
    enc_arr = (enc.fit_transform(df[['Education']])).astype(int)
    
    return enc_arr

# generate features
def get_features(df_train_imp):
    """
    Input
    df_train_imp: dataframe with imputed values
    Output
    df_train_imp3: dataframe with generated features
    """
    
    # check if required variables exist
    a = {'CallEnd', 'CallStart', 'Job', 'Marital', 'Education', 'LastContactMonth', 'Outcome'}
    b = set(df_train_imp.columns.to_list())
    if a.issubset(b):
        
        # job
        df_job = pd.get_dummies(data = df_train_imp['Job'], prefix = "Job")

        # martial
        df_marital = pd.get_dummies(data = df_train_imp['Marital'], prefix = "Marital")

        # education - call encoder
        #df_ed = pd.get_dummies(data = df_train_imp['Education'], prefix = "Education")
        df_train_imp['EducationEncoded'] = encode_ed(df_train_imp)

        # last contact month
        df_lcm = pd.get_dummies(data = df_train_imp['LastContactMonth'], prefix = "LastContactMonth")

        # outcome
        df_outcome = pd.get_dummies(data = df_train_imp['Outcome'], prefix = "Outcome")

        # calculate call duration
        # unit test: from test_transforms import test_calc_call_duration
        df_train_imp2 = calc_call_duration(df_train_imp)

        # drop cols where one hot encoding was used
        df_train_imp2.drop(['Outcome', 'LastContactMonth', 'Education', 'Marital', 'Job'], axis = 1, inplace = True)

        # merge in the one hot dataframes
        df_train_imp3 = pd.concat([df_train_imp2, df_job, df_marital, df_lcm, df_outcome], axis=1)
        print(df_train_imp3.shape)
        
    else:
        raise KeyError("Required fields are missing for feature engineering")
    
    return df_train_imp3

# oversample function to create a class balance
def oversample(df):
    """
    Input
    df: dataframe with class imbalance in CarInsurance
    Output
    final_df: dataframe with balanced class in CarInsurance
    """
    
    classes = df.CarInsurance.value_counts().to_dict()
    most = max(classes.values())
    
    # define var to be balanced
    classes_list = []
    for key in classes:
        classes_list.append(df[df['CarInsurance'] == key]) 
    classes_sample = []
    
    # pick sample and append to data
    for i in range(1,len(classes_list)):
        classes_sample.append(classes_list[i].sample(most, replace=True))
    df_maybe = pd.concat(classes_sample)
    final_df = pd.concat([df_maybe,classes_list[0]], axis=0)
    final_df = final_df.reset_index(drop=True)
    return final_df