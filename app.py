from src.insurance_charge_predict.logger import logging
from src.insurance_charge_predict.exception import CustomException  # ✅ spelling fix
import sys
from src.insurance_charge_predict.components.data_ingestion import DataIngestion
from src.insurance_charge_predict.components.data_transformation import DataTransformationConfig, DataTransformation
from src.insurance_charge_predict.components.model_trainer import ModelTrainerConfig, ModelTrainger
#api ke liye 

if __name__ == "__main__":
    logging.info("The execution has started....")
    # data ingestion
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    #data transformatioon
        data_transformation=DataTransformation()
        train_arr, test_arr,_= data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    # model Training

        model_trainer=ModelTrainger()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)  # ✅ spelling fix
    
    
