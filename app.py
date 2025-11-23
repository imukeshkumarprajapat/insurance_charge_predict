from src.insurance_charge_predict.logger import logging
from src.insurance_charge_predict.exception import CustomException  # ✅ spelling fix
import sys
from src.insurance_charge_predict.components.data_ingestion import DataIngestion
from src.insurance_charge_predict.components.data_transformation import DataTransformationConfig, DataTransformation
from src.insurance_charge_predict.components.model_trainer import ModelTrainerConfig, ModelTrainger
from flask import Flask, render_template, request

from src.insurance_charge_predict.pipelines.predication_pipeline import CustomData, PredictionPipline
import dagshub
#api ke liye 


dagshub.init(repo_owner='imukeshkumarprajapat', repo_name='insurance_charge_predict', mlflow=True)

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
    
    


# import pandas as pd

# # Flask app initialize


# from flask import Flask, request, render_template

# app = Flask(__name__)   # yahan pp ki jagah app rakho


# @app.route('/')
# def index():
#     return render_template('index.html') 

# @app.route("/predictdata", methods=['GET','POST'])
# def predict_datapoint():
#     if request.method == "GET":
#         return render_template("home.html")
#     else:
#         # CustomData object banate waqt arguments pass karo
#         data = CustomData(
#             age=int(request.form.get('age')),
#             sex=request.form.get('sex'),
#             bmi=float(request.form.get('bmi')),
#             children=int(request.form.get('children')),
#             smoker=request.form.get('smoker'),
#             region=request.form.get('region')
#         )

#         pred_df = data.get_data_as_data_frame()
#         print(pred_df)
#         print("Before Prediction")

#         predict_pipeline = PredictionPipline()
#         print("Mid Prediction")
#         results = predict_pipeline.predict(pred_df)
#         print("After Prediction")

#         return render_template('home.html', results=results[0])

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", debug=True)



