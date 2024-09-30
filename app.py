#!/usr/bin/env python
# coding: utf-8

# In[18]:


pip install fastapi uvicorn nest_asyncio scikit-learn pandas pydantic lightgbm


# In[19]:


pip install --user scikit-learn==1.2.2


# In[43]:


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model = joblib.load('model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

feature_order = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 
    'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
]

class InputData(BaseModel):
    job: int
    marital: int
    education: int
    default: int
    housing: int
    loan: int
    month: int
    day_of_week: int
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: int

class DataFrameInput(BaseModel):
    data: List[InputData]

def preprocess_input(data: DataFrameInput, label_encoders, scaler):
    df = pd.DataFrame([item.dict() for item in data.data])
    logger.debug(f"Initial DataFrame: {df.head()}")
    
    missing_columns = set(feature_order) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in input data: {missing_columns}")

    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'month', 'day_of_week', 'poutcome']
    for column in categorical_columns:
        if column in df.columns:
            le = label_encoders.get(column, None)
            if le:
                df[column] = df[column].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[column] = le.transform(df[column])
            else:
                raise ValueError(f"Label encoder for column '{column}' not found")

    df = df[feature_order]
    
    logger.debug(f"DataFrame before scaling: {df.head()}")

    numerical_features = ['duration', 'campaign', 'pdays', 'previous']
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    logger.debug(f"DataFrame after scaling: {df.head()}")
    return df

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}

@app.post("/predict")
def predict(data: DataFrameInput):
    try:
        if not data.data:
            raise HTTPException(status_code=400, detail="Input data must contain at least one item.")
        
        processed_data = preprocess_input(data, label_encoders, scaler)

        logger.debug(f"DataFrame before prediction: {processed_data.head()}")

        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)[:, 1]
        
        return {"prediction": prediction.tolist(), "prediction_proba": prediction_proba.tolist()}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# In[ ]:


# Langsung run uvicornnya
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

