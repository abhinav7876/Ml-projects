import os
import sys

import numpy as np 
import pandas as pd
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        for k,model in models.items():
            para=param[k]
            gs=GridSearchCV(model,para,cv=5)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            r2score=r2_score(y_test,y_pred)
            report[k]=r2score
        return report
    except Exception as e:
        raise CustomException(e,sys)
def load_object(file_path):
    try:
        with open(file_path,"rb") as obj:
            return pickle.load(obj)
    except Exception as e:
        raise CustomException(e,sys)


