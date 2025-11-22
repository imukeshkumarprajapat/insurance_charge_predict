# import sys
# from dataclasses import dataclass
# from src.insurance_charge_predict.exception import CustomeException
# from src.insurance_charge_predict.logger import logging
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from src.insurance_charge_predict.utils import save_object
# from src.insurance_charge_predict.logger import logging
# import os


# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config=DataTransformationConfig()
    
#     # def get_data_transformer_object(self):
#     #     """this function is responsible for data transformation"""
#     #     try:
#     #         numerical_columns=["age","is_female","bmi","children","region_southeast","bmi_category_Obese"]
#     #         #ye pipeline sirf numeric features ke liye h agr categorical data ho to uske liye alg se bnegi
#     #         num_pipeline=Pipeline(steps=[
#     #         ("imputer",SimpleImputer(strategy="median")), #missing value ko handle karna update data aye tb
#     #         ('scaler',StandardScaler())

#     #         ])
#     #         logging.info(f"numerical columsn:{numerical_columns}")
#     #=====================================================================

#     def get_data_transformer_object(self, df):
#         try:
#         # Detect categorical and numerical columns dynamically
#             cat_columns = df.select_dtypes(include="object").columns
#             num_columns = df.select_dtypes(exclude="object").columns

#         # Numerical pipeline
#             num_pipeline = Pipeline(steps=[
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler())
#             ])

#         # Categorical pipeline (only if categorical columns exist)
#             cat_pipeline = Pipeline(steps=[
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("onehot", OneHotEncoder(handle_unknown="ignore"))
#              ]) if len(cat_columns) > 0 else None
#             logging.info(f"numerical columns:{num_pipeline},{cat_pipeline}")
#             # Preprocessor=ColumnTransformer([
#             #     ("num_pipline", num_pipeline, numerical_columns)
#             # ])  

#             # return Preprocessor
        

#             # ColumnTransformer
#             transformers = []
#             transformers.append(("num_pipeline", num_pipeline, num_columns))
#             if cat_pipeline:
#                transformers.append(("cat_pipeline", cat_pipeline, cat_columns))

#             preprocessor = ColumnTransformer(transformers)
#             return preprocessor

#         except Exception as e:
#             raise CustomeException(e, sys)


        

# #Pipeline ko fit karna sirf training data par hota hai
# #Jab hum scaler, encoder, imputer use karte hain, unke parameters (mean, std, categories, median, etc.) sirf training data se learn hote hain.
# #Agar test data se bhi parameters learn karenge to data leakage ho jaayega (model ko unseen data ke baare mein pehle se knowledge mil jaayegi).

#     def initiate_data_transformation(self, train_path, test_path):
       
#         try:
#             train_df=pd.read_csv(train_path)
#             test_df=pd.read_csv(test_path)
            


#             logging.info("Reading the train and test file")
#             preprocessing_obj=self.get_data_transformer_object(train_df)
            
             
#             target_column_name = "charges"
#             X_train = train_df.drop(columns=[target_column_name], axis=1)
#             y_train = train_df[target_column_name]


#             X_test = test_df.drop(columns=[target_column_name], axis=1)
#             y_test = test_df[target_column_name]
#             #target_columns_name="charges"
#             #numerical_columns=["age","is_female","bmi","children","region_southeast","bmi_category_Obese"]
             
#              #divid the train dataset to independent and dependent feature

            


#             #input_features_train_df=train_df.drop(columns=[target_columns_name], axis=1)
#             #target_features_train_df=train_df[target_columns_name]

# ## divid the test dataset to independent and dependent feature

#             #input_features_test_df=test_df.drop(columns=[target_columns_name], axis=1)
#             #target_features_test_df=test_df[target_columns_name]
             
#             logging.info("Applying Preprocessing on training and test dataframe")
#             #input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
#             #input_feature_test_arr=preprocessing_obj.transform(input_features_test_df)
#             input_feature_train_arr = preprocessing_obj.fit_transform(X_train)
#             input_feature_test_arr = preprocessing_obj.transform(X_test)


#         #     train_arr=np.c_[
#         #         input_feature_train_arr, np.array(target_features_train_df)
#         #     ]
#         #     test_arr=np.c_[
#         #         input_feature_test_arr, np.array(target_features_test_df)
#         #     ]
#         #     logging.info(f"Save preprocessing object")

#         #     save_object(
#         #         file_path=self.data_transformation_config.preprocessor_obj_file_path,
#         #         obj=preprocessing_obj
#         #     )
#         #     return train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path


#         # except Exception as e:
#         #     raise CustomeException(sys, e)



#              # Combine features + target
#             train_arr = np.c_[input_feature_train_arr, np.array(y_train)]
#             test_arr = np.c_[input_feature_test_arr, np.array(y_test)]

#             logging.info("Saving preprocessing object")

#             save_object(
#             file_path=self.data_transformation_config.preprocessor_obj_file_path,
#             obj=preprocessing_obj
#             )

#             return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

#         except Exception as e:
#             raise CustomeException(e, sys)



import sys
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.insurance_charge_predict.exception import CustomException
from src.insurance_charge_predict.logger import logging
from src.insurance_charge_predict.utils import save_object

# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()
    
#     def get_data_transformer_object(self, df):
        
#         try:
#              # Detect categorical and numerical columns dynamically
#             cat_columns = df.select_dtypes(include="object").columns
#             num_columns = df.select_dtypes(exclude="object").columns
     
#         # Numerical pipeline
#             num_pipeline = Pipeline(steps=[
#                 ("imputer", SimpleImputer(strategy="median")),
#                 ("scaler", StandardScaler())
#              ])
     
#              # Categorical pipeline (only if raw categorical columns exist)
#             cat_pipeline = None
#             if set(["sex", "smoker", "region"]).intersection(df.columns):
#                 cat_pipeline = Pipeline(steps=[
#                      ("imputer", SimpleImputer(strategy="most_frequent")),
#                      ("onehot", OneHotEncoder(handle_unknown="ignore"))
#                  ])

#             logging.info(f"Numerical columns: {list(num_columns)}")
#             logging.info(f"Categorical columns: {list(cat_columns)}")

#             transformers = []
#             transformers.append(("num_pipeline", num_pipeline, num_columns))
#             if cat_pipeline:
#                 transformers.append(("cat_pipeline", cat_pipeline, cat_columns))

#             preprocessor = ColumnTransformer(transformers)
#             return preprocessor

#         except Exception as e:
#             raise CustomeException(e, sys)
    


#     def initiate_data_transformation(self, train_path, test_path):
       
#         try:
#             train_df=pd.read_csv(train_path)
#             test_df=pd.read_csv(test_path)
            


#             logging.info("Reading the train and test file")
#             preprocessing_obj=self.get_data_transformer_object(train_df)
            
             
#             target_column_name = "charges"
#             X_train = train_df.drop(columns=[target_column_name], axis=1)
#             y_train = train_df[target_column_name]


#             X_test = test_df.drop(columns=[target_column_name], axis=1)
#             y_test = test_df[target_column_name]
            
             
#             logging.info("Applying Preprocessing on training and test dataframe")
            
#             input_feature_train_arr = preprocessing_obj.fit_transform(X_train)
#             input_feature_test_arr = preprocessing_obj.transform(X_test)



#              # Combine features + target
#             train_arr = np.c_[input_feature_train_arr, np.array(y_train)]
#             test_arr = np.c_[input_feature_test_arr, np.array(y_test)]

#             logging.info("Saving preprocessing object")

#             save_object(
#             file_path=self.data_transformation_config.preprocessor_obj_file_path,
#             obj=preprocessing_obj
#             )

#             return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

#         except Exception as e:
#             raise CustomeException(e, sys)

import sys
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.insurance_charge_predict.exception import CustomException
from src.insurance_charge_predict.logger import logging
from src.insurance_charge_predict.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, df):
        try:
            # Check if raw dataset (sex, smoker, region present)
            if set(["sex", "smoker", "region"]).issubset(df.columns):
                logging.info("Raw dataset detected")
                num_columns = ["age", "bmi", "children"]
                cat_columns = ["sex", "smoker", "region"]

                num_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])

                cat_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ])

                preprocessor = ColumnTransformer([
                    ("num_pipeline", num_pipeline, num_columns),
                    ("cat_pipeline", cat_pipeline, cat_columns)
                ])

            else:
                # Clean dataset mode (engineered features already numeric)
                logging.info("Clean dataset detected")
                num_columns = [col for col in df.columns if col != "charges"]

                num_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])

                preprocessor = ColumnTransformer([
                    ("num_pipeline", num_pipeline, num_columns)
                ])

            logging.info(f"Preprocessor created with columns: {preprocessor.transformers}")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj = self.get_data_transformer_object(train_df)

            target_column_name = "charges"

            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(X_train)
            input_feature_test_arr = preprocessing_obj.transform(X_test)

            # Combine features + target
            train_arr = np.c_[input_feature_train_arr, np.array(y_train)]
            test_arr = np.c_[input_feature_test_arr, np.array(y_test)]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)