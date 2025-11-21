from src.insurance_charge_predict.logger import logging
from src.insurance_charge_predict.exception import CustomeException  # ✅ spelling fix
import sys
from src.insurance_charge_predict.components.data_ingestion import DataIngestion
from src.insurance_charge_predict.components.data_transformation import DataTransformationConfig, DataTransformation
#api ke liye 

if __name__ == "__main__":
    logging.info("The execution has started....")
    # data ingestion
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()


        data_transformation=DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomeException(e, sys)  # ✅ spelling fix
    
    # data transformation
