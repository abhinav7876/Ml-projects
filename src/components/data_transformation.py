import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


from src.utils import save_object
@dataclass
class DataTransformation_config:
    preprocessor_obj_path=os.path.join("artifacts",'preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformation_config()
    def data_transformation_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                    "gender",
                    "race_ethnicity",
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course",
                ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),#using meadian as outliers are there
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),#using meadian as outliers are there
                    ("One Hot Encoder",OneHotEncoder())
                ]
            )
            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
            ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            df_train=pd.read_csv(train_path)
            df_test=pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.data_transformation_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=df_train.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=df_train[target_column_name]

            input_feature_test_df=df_test.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=df_test[target_column_name]

            logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe."
                )
            processed_train_df=preprocessing_obj.fit_transform(input_feature_train_df)
            processed_test_df=preprocessing_obj.transform(input_feature_test_df)
            train_arr=np.c_[processed_train_df,np.array(target_feature_train_df)]#combining two 1D array into one 2D array
            test_arr=np.c_[processed_test_df,np.array(target_feature_test_df)]
            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj)
            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_path)
        except Exception as e:
            raise CustomException(e,sys)



        
