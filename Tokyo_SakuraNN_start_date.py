# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:03:38 2022

@author: annam
"""
from Tokyo_SakuraNN import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
#%%

class Seasons_by_Year(pd.DataFrame):   
    def __init__(self, year):
        pd.DataFrame.__init__(self)
        self.year = year
        super().__init__(pd.DataFrame(sakura[sakura["year"]==str(year)].reset_index())) 


def make_seasons_(start, start_year, end_year):
    start =  start
    seasons = [] 
    dict_index_year = make_dict_index_year(start_year, end_year)
    dict_year_index = make_dict_year_index(start_year, end_year)
    n_years = end_year-start_year
    print(n_years)
    for i in range(0,n_years):       
        season_new = Seasons_by_Year(dict_index_year[i])
        print(season_new)
        v1 =season_new.index[season_new.date == start.iloc[i, 1]].tolist() # von
      
        season_new.loc[v1[0]: ,"flower_status"] =1

        season_new["temp_cs"] = season_new["temperature"].cumsum()

        seasons.append(season_new)
    return seasons    
 

def prepare_features_tokyo(sakura, start_year, end_year):
    start_full_scatter = make_start_full_scatter(sakura)
    start = start_full_scatter["start"]

    seasons = make_seasons_(start, start_year, end_year)
    
    features_10 =[]                
    for i in range(0,len(seasons)):
        feature =seasons[i].drop(['index', 'date', 'year',"flower_status"],axis='columns',inplace=False).reset_index(drop=True)
        feature.loc[10:,"temperature"]=0
        feature.loc[10:,"temp_cs"]=0
        features_10.append(feature)
        
    features_10_new = features_10[0] 
    for i in range(1,len(features_10)):
        feature_i = features_10[i]
        features_10_new =features_10_new.append(feature_i)
    return features_10_new

def make_labels_tokyo(sakura, start_year, end_year):
    start_full_scatter = make_start_full_scatter(sakura)
    start = start_full_scatter["start"]

    seasons = make_seasons_(start, start_year, end_year)

    
    s_all = seasons[0]
    for i in range(1,len(seasons)):
        s_i = seasons[i]
        s_i["temp_cs"]=s_i["temperature"].cumsum()
        s_all =s_all.append(s_i)

    labels = s_all.drop(['index', 'date', 'year', "flower_status",'month','day', "temp_cs"],axis='columns',inplace=False).reset_index(drop=True) # f??r len = 90 output
    
    return labels


def make_model_nn_():
    model_nn = keras.Sequential([
	keras.layers.Dense(210, input_shape=(4,), activation='relu'), # day, month, temp, temp_cs
    keras.layers.Dropout(0.3),
	keras.layers.Dense(420, activation='relu'),
    keras.layers.Dropout(0.3),
	keras.layers.Dense(420, activation='relu'),
    keras.layers.Flatten(),
	keras.layers.Dense(2, activation='sigmoid')]) # 0,1 one-hot encoded
   
    model_nn.compile(optimizer="adam", 
	          loss="categorical_crossentropy",
	          metrics=['accuracy'])
    return model_nn

def plot_confusion_matrix(cm, class_names, title):
   
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    

    threshold = cm.max()*9 / 10
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("cm_start.png")
    return figure

def show_cm(data):
    X_test = data["X_test"]
    y_test = data["y_test"]
    rounded_prediction = np.argmax(model_nn.predict(X_test).round(2), axis = 1)# 


    rounded_y_test = np.argmax(y_test, axis = 1)

    cm = confusion_matrix(y_true=rounded_y_test, y_pred = rounded_prediction) 

    cm_plot_labels = ["0", "1"]
    
    plot_confusion_matrix(cm=cm, class_names = cm_plot_labels,title = "Confusion Matrix")
#%%

if __name__ == '__main__':
    path = os.path.abspath('tokyo_sakura.csv')
    sakura = load_and_clean_data(path)

    start_year = 2010
    end_year = 2019
    start_full_scatter = make_start_full_scatter(sakura)
    seasons = make_seasons_(start_full_scatter["start"], start_year, end_year)    

    data = make_train_test(make_features_nn(seasons), make_labels_nn(seasons), 0.3)    
    model_nn = make_model_nn_()
    trained_nn = train_model_nn(model_nn, data, 10, 100)
    show_cm(data)
    save_net(trained_nn, "Tokyo_NN_start")
    print("Trained net was saved as Tokyo_NN_start")
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
