print("main.py is being executed")

import joblib 
import warnings
from fastapi import FastAPI
from Star_Properties import StarInput
from Star_types_data import star_types

# Ignore warnings due to version changes
warnings.filterwarnings('ignore')

# Create FastAPI application
app = FastAPI()

#Load the pipeline
path_pipeline = 'star_pred.joblib'
pipeline = joblib.load(path_pipeline)

@app.get("/")
def read_root():
    return {"message": "App running!"}

#Get the predictions using post request
@app.post("/predict")
def prediction(input_data:StarInput):
    #Get the data
    data = [[
        input_data.temperature,
        input_data.luminosity,
        input_data.radius,
        input_data.absolute_magnitude
    ]]

    #make predictions
    prediction = pipeline.predict(data)[0]
    confidence_score = pipeline.predict_proba(data)[0][prediction]
    print(confidence_score)
    return {
            'Predicted star type': star_types[prediction],
            'Confidence_score': str(round(confidence_score*100,1))+ "%"}


