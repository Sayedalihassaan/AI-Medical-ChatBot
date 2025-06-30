import os
from pathlib import Path
import logging




list_of_files = [
    ".env" , 
    ".env.example" , 
    "requirements.txt" , 
    "README.md" , 
    "LICENSE" , 
    "frontend.py" , 
    "setup.py" , 
    "Notebook/AI_Medical_ChatBot.ipynb" , 
    "System/helper.py" , 
    "System/strcture.py" , 
    "store_db.py" , 
    "DataSets"
]

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")


for filepath in list_of_files:
     filepath = Path(filepath)
     filedir, filename = os.path.split(filepath)

     if filedir != "":
          os.makedirs(filedir, exist_ok=True)
          logging.info(f"Creating Directory {filedir} for the file name {filename}")
          
     if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0 ):
          with open(file=filepath, mode="w") as f:
               pass
               logging.info(f"Creating empty file: {filepath}")
               
     else:
          logging.info(f"{filepath} is already exists")
