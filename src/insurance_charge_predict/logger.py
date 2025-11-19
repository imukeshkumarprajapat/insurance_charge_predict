import os
import logging
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"logs", LOG_FILE) #logfile ka path set kiya h yaha pr 
os.makedirs(log_path,exist_ok=True)



LOG_FILE_PATH=os.path.join(log_path, LOG_FILE) #dono ko combine kar diya


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",level=logging.INFO
    #jo bi loggin file me show hoga usko is fomate ke hisab se msg show karna h 
)