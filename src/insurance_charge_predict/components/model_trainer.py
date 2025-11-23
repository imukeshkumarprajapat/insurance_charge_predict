import os
import sys
import numpy as np
from dataclasses import dataclass
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import mlflow

import mlflow.sklearn
from src.insurance_charge_predict.exception import CustomException
from src.insurance_charge_predict.logger import logging
#from src.insurance_charge_predict import save_object,evaluate_models

from src.insurance_charge_predict.utils import save_object, evalute_models




from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainger:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def eval_metics(self, actual, pred):
        rmse=np.sqrt(mean_squared_error(actual, pred))
        mae=mean_absolute_error(actual, pred)
        r2=r2_score(actual, pred)

        return rmse, mae, r2
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
    train_array[:, :-1],                 # features → all columns except last
    train_array[:, -1].reshape(-1, 1),   # target → reshape to column vector
    test_array[:, :-1],                  # features → all columns except last
    test_array[:, -1].reshape(-1, 1)     # target → reshape to column vector
)
            
            #optional
           # X_train = train_df.drop("charges", axis=1)
           # y_train = train_df[["charges"]]   # double brackets → DataFrame (2D)

            #X_test = test_df.drop("charges", axis=1)
            #y_test = test_df[["charges"]]
           

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor()
            }
            #hyperparameter tunnig key value parameters

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [100,200],
                    'max_depth':[None,10,20]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "KNeighborsRegressor":{
                     "n_neighbors": [5]

                },


                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report: dict = evalute_models(X_train, y_train, X_test, y_test, models, params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
# to get best model name from dict
            # best_model_name=list(model_report.keys())[
            #     list(model_report.values().index(best_model_score))
            # ]
            best_model_score = max(model_report.values())   # सबसे अच्छा score निकालो
            best_model_index = list(model_report.values()).index(best_model_score)
            best_model_name  = list(model_report.keys())[best_model_index]



            best_model=models[best_model_name]


        
            print("this is the best model:")
            print(best_model_name)



            model_names=list(params.keys())

            actual_model=" "

            for model in model_names:
                if best_model_name==model:
                    actual_model=actual_model+model
            
            best_params=params[actual_model.strip()]
            print(f"Model selected: '{actual_model}'")

            print("\nAll Model Results:")
            print("Model Name\t\tR2_Score")
            for name, score in model_report.items():
              print(f"{name:25} {score:.6f}")



            mlflow.set_tracking_uri("https://dagshub.com/imukeshkumarprajapat/insurance_charge_predict.mlflow")
            mlflow.set_experiment("Default")
            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

            #mlflow track
            with mlflow.start_run():
                predicted_qualities=best_model.predict(X_test)
                (rmse,mae,r2)=self.eval_metics(y_test, predicted_qualities)
                #mlflow.log_param(best_model)
                mlflow.log_param("best_model", best_model_name.strip())  # ✅ Correct
                mlflow.log_metric("rmse",rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)
                #model registery does not work with file store
                
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(best_model, artifact_path="model")
                #mlflow.sklearn.log_model(best_model, "model")
               # mlflow.sklearn.save_model(best_model, path="artifacts/best_model")
                
                
            else:
                mlflow.sklearn.log_model(best_model, "model")

            






            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,obj=best_model
            )


            predicted=best_model.predict(X_test)
            
            r2_square=r2_score(y_test, predicted)

            return r2_square



        except Exception as e:
            raise CustomException(e,sys)