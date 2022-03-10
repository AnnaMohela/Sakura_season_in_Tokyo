# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:59:04 2022

@author: annam
"""
import pandas as pd
import numpy as np
from tensorflow import keras
from Tokyo_SakuraNN import make_dict_year_index
from sklearn.preprocessing import MinMaxScaler
import os

#%%
def make_index_date_dictionary():
    calender_dates =[]

    for i in range(1,32):
        date = str(i)+".3"
        calender_dates.append(date)
    for i in range(1,31):
        date = str(i)+".4"
        calender_dates.append(date)
    for i in range(1,32):
        date = str(i)+".5"
        calender_dates.append(date)
 
    index_date =dict(zip(range(0,92), calender_dates))
    return index_date
    
def predicted_dates(prediction): # Korrekturterme in return
    start_id = []
    full_id =[]
    scatter_id = []
    for i in range(0,len(prediction)):
        if prediction[i,1]>prediction[i,0] and prediction[i,1]>prediction[i,2]:
            start_id.append(i)

        if (prediction[i,2]>prediction[i,1]and prediction[i,2]>prediction[i,3]) or prediction[i,2]>0 and prediction[i,3]<1:
            full_id.append(i)
            continue

        if (prediction[i,3]>0 and prediction[i,2]<prediction[i,3]):
            scatter_id.append(i)

    index_date = make_index_date_dictionary()

    return index_date[start_id[0]], index_date[full_id[int(len(full_id)/2)]], index_date[scatter_id[0]]      
 

def predicted_start_date(prediction): 
    start_id = []
    for i in range(0,len(prediction)):
        if prediction[i,1]>prediction[i,0]:
            start_id.append(i)

    index_date = make_index_date_dictionary()

    return index_date[start_id[0]]

#%% # make prediction      
    
    
def predict_dates_LRnn(X, model_temp, model_nn, only_start):
    X = np.asarray(X).astype("float32")

    prediction_LR = model_temp.predict(X).round(2) 
    prediction_LR_df = pd.DataFrame(prediction_LR)
    input_nn = pd.read_pickle("dummy_df.pkl")
    input_nn["temperature"] = prediction_LR_df[0]
    input_nn["temp_cs"]=input_nn["temperature"].cumsum() 

    input_nn=np.asarray(input_nn).astype("float32") 
    input_nn =MinMaxScaler().fit_transform(input_nn) 
    prediction_LRnn = model_nn.predict(input_nn).round(2) 

    if only_start:
        dates_LRnn = predicted_start_date(prediction_LRnn)
    else:
        dates_LRnn =predicted_dates(prediction_LRnn) 
    return dates_LRnn

def get_data_for_prediction(path_data, n_days,year): 
    dict_year_index = make_dict_year_index(2010,2020)
    i = dict_year_index[year]
    temp_ = pd.read_csv(path_data)

    temp_data=temp_.iloc[92*i:92*(i+1)].reset_index(drop=True)

    temp_data[['year', 'month','day']] = temp_data['date'].str.rsplit('/', 2, expand=True)
    temp_data["temp_cs"]=temp_data["temperature"].cumsum()
    temp_data["temperature"]=temp_data["temperature"].fillna(0)
    temp_data["temp_cs"]=temp_data["temp_cs"].fillna(0)  
    temp_data.drop(["year","date", "flower_status"],axis='columns',inplace=True)
    temp_data.loc[n_days:,"temperature"]=0     
    temp_data.loc[n_days:,"temp_cs"]=0 
    
    return temp_data

def get_data_for_actual_prediction(path_data):
    temp_data = pd.read_csv(path_data)
    
    temp_data[['year', 'month','day']] = temp_data['date'].str.rsplit('/', 2, expand=True)
    temp_data["temp_cs"]=temp_data["temperature"].cumsum()
    temp_data["temperature"]=temp_data["temperature"].fillna(0)
    temp_data["temp_cs"]=temp_data["temp_cs"].fillna(0)  
    temp_data.drop(["year","date"],axis='columns',inplace=True)
    
    return temp_data


def make_prediction(path_data, n_days, year, test_yes_no, only_start):
    if n_days==10:
        model_path = os.path.abspath('GBreg_retrained_10.sav')
    else:
        model_path = os.path.abspath('GB_model_30.sav')
    model_temp = pd.read_pickle(model_path)
    if only_start==True:
        model_nn = os.path.abspath('Tokyo_NN_start.h5')
    else:
        model_nn = os.path.abspath('Tokyo_NN.h5')
    if test_yes_no==True:
        temp_data = get_data_for_prediction(path_data, n_days, year) # test with old data
    else:
        temp_data= get_data_for_actual_prediction(path_data)
        
    prediction = predict_dates_LRnn(temp_data, model_temp, model_nn, only_start)
    return prediction
#%% 


if __name__ == '__main__':
    
    path=os.path.abspath('tokyo_2022.csv')
    only_start_yes_no = input("Predict only the start? (Y,N): ")
    if only_start_yes_no == "Y":
        only_start=True
    else: 
        only_start=False
    prediction = make_prediction(path, 10, 2022, False, only_start) 
    if only_start:
        print("Predicted start date this year:")
    else:
        print("Predicted dates for this year:")
    print(prediction)
    print()

