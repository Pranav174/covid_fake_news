from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import tensorflow as tf
from uvicorn import run
import tensorflow_text 
import os

app = FastAPI()

covid_fake_news_model = load_model('covid_fake_news_model')

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Food Vision API!"}

@app.get("/search")
async def search(query:str):
    predictions = covid_fake_news_model.predict([query])
    score = tf.nn.softmax(predictions,axis=-1).numpy()[0][0]
    return {"score": score.item()}
    
if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	run(app, host="0.0.0.0", port=port)