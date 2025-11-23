import os 
import sys

from src.insurance_charge_predict.exception import CustomException
from src.insurance_charge_predict.logger import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV
from dotenv import load_dotenv
import pymysql
from sklearn.metrics import r2_score
import pickle
import numpy as np

load_dotenv()  #load .env 

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")


def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection established: %s", mydb)
        df = pd.read_sql_query("SELECT * FROM insurance", mydb)
        print(df.head())
        return df  # ✅ This was missing!
    except Exception as ex:
        raise CustomException(ex, sys)
    


def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)


    except Exception as ex:
        raise CustomException(ex, sys)  # ✅ Also pass sys for traceback
    

# from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import sys

# def evalute_models(X_train, y_train, X_test, y_test, models, params):
#     try:
#         report = {}
#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             para  = params[list(models.keys())[i]]

#             gs = GridSearchCV(model, para, cv=3)
#             gs.fit(X_train, y_train)

#             # set best params
#             model.set_params(**gs.best_params_)
#             model.fit(X_train, y_train)

#             # predictions
#             X_train_pred = model.predict(X_train)
#             X_test_pred  = model.predict(X_test)

#             # ✅ Correct order of r2_score
#             train_model_score = r2_score(y_train, X_train_pred)
#             test_model_score  = r2_score(y_test, X_test_pred)

#             report[list(models.keys())[i]] = test_model_score

#         return report

#     except Exception as e:
#         raise CustomException(e, sys)

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import sys
from src.insurance_charge_predict.exception import CustomException

def evalute_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        
        # ✅ Loop directly over key, value pairs (no index juggling)
        for key, model in models.items():
            para = params[key]   # get parameter grid for this model

            # GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train.ravel())   # ✅ y_train reshaped to 1D

            # set best params and retrain
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train.ravel())

            # predictions
            X_train_pred = model.predict(X_train)
            X_test_pred  = model.predict(X_test)

            # ✅ Correct order of r2_score
            train_model_score = r2_score(y_train, X_train_pred)
            test_model_score  = r2_score(y_test, X_test_pred)

            # save test score in report
            report[key] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        print("Trying to load model from:", file_path)

        with open(file_path,"rb") as file_obj:
             return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)