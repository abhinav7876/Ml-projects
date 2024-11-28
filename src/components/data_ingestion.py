import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformation_config
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import Model_trainer_config
from dataclasses import dataclass

@dataclass
class data_ingestion_config:  #basically has all the components required for ingestion
    train_path:str =os.path.join("artifacts","train.csv") #no need to use init for class variable as we are using decorator for class dataclass
    test_path:str =os.path.join("artifacts","test.csv")
    raw_path:str =os.path.join("artifacts","raw.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config=data_ingestion_config()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("notebook/data/stud.csv")
            logging.info("data set readed from source")

            os.makedirs((os.path.dirname(self.ingestion_config.raw_path)),exist_ok=True) #artifacts folder created
            df.to_csv(self.ingestion_config.raw_path,index=False,header=True)
            logging.info("Main data file created in artifacts folder")

            Train,test=train_test_split(df,test_size=0.2,random_state=42)
            logging.info("Data splitted into train test")

            Train.to_csv(self.ingestion_config.train_path,index=False,header=True)
            logging.info("Train data file created in artifacts folder")

            test.to_csv(self.ingestion_config.test_path,index=False,header=True)
            logging.info("Test data file created in artifacts folder")

            logging.info("Data ingestion completed")
            return (self.ingestion_config.train_path,self.ingestion_config.test_path)
        except Exception as e:
            raise Custom_exception(e,sys)
'''
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
'''
if __name__=="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    transformer_obj=DataTransformation()
    X_arr,y_arr,_=transformer_obj.initiate_data_transformation(train_path,test_path)
    model_trainer_obj=ModelTrainer()
    print(model_trainer_obj.model_trainer(X_arr,y_arr))
 
    