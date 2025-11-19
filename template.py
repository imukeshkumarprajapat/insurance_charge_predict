import os
from pathlib import Path
import logging



logging.basicConfig(level=logging.INFO)

project_name="isurance_charge_pridict"

list_of_files=[
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitering.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipelines.py",
    f"src/{project_name}/pipelines/predication_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files: #ye ek list h isme file ka paths (strings) me diye gye or us path pr loop chlega  
    filepath=Path(filepath)   #- Path class (from pathlib) ka use karke string ko ek Path object me convert kiya ja raha hai.
    filedir, filename=os.path.split(filepath)

    if filedir != "": #"Agar filedir khaali string nahi hai (yaani koi folder ka naam diya gaya hai), tab hi folder banao."

        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")


    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0): #agr path nhi milta h ya file ka size 0 h to pass kar do 
        with open(filepath, 'w')as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exits")
