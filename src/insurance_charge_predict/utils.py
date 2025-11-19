import os 
import sys

from src.insurance_charge_predict.exception import CustomeException
from src.insurance_charge_predict.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

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
        raise CustomeException(ex, sys)  # ✅ Also pass sys for traceback