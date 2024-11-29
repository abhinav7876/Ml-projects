import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
class PredictPipeline:
    def __init__(self):
        pass
    def predict_data(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            preprocessor=load_object(preprocessor_path)
            logging.info("preprocessor loaded")
            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(model_path)
            logging.info("model loaded")

            preprocessed_data=preprocessor.transform(features)
            logging.info("preprocessing done")
            prediction=model.predict(preprocessed_data)
            logging.info("model prediction done")

            return prediction
        except Exception as e:
            raise CustomException(e,sys)
class custom_data:
    def __init__(self,
                gender: str,
                race_ethnicity: str,
                parental_level_of_education,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    def data_to_df(self):
        try:
            data_dict={
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e,sys)