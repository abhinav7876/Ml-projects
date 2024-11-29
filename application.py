from flask import Flask,render_template,request
from src.utils import save_object
from src.pipeline.predict_pipeline import PredictPipeline,custom_data
from src.logger import logging

application=Flask(__name__)
app=application
@app.route('/')# Route for a home page
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=custom_data(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )
        df=data.data_to_df()
        print("Input data")
        print(df)
        predict_obj=PredictPipeline()
        results=predict_obj.predict_data(df)
        logging.info("returned final prediction result")
        return render_template("home.html",results=results[0])#as it will be in list
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)



