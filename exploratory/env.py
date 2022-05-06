# script to store environment parameters 
import os

# set up logger
logger_path = r'../log/notebook_log.txt'
# delete old log if exists
if os.path.exists(logger_path):
    os.remove(logger_path)

# path to raw data
train_data = r'../data/carInsurance_train.csv'
test_data = r'../data/carInsurance_test.csv'

# data folder
data_path = r'../data/'

# path to model folder
model_path = r'../model/'