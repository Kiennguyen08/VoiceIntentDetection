from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import *
from werkzeug.utils import secure_filename
import json
import numpy as np
import model
import os
from tensorflow import keras
from modelVoice import *
import os
import shutil
import requests
# Creating FastAPI instance
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = keras.models.load_model('./weights/TFIDF-keras.h5')
tfidf_pickle = load_tfidf_vectorizer()

@app.get("/check")
def checkServer():
    return {"server":"up"}

@app.post("/addData/")
async def create_upload_file(file: UploadFile = File(...), username: str = Form(...)):
    print("USERNAME",username)
    print("File", file.filename)
    for f in os.listdir('./uploads_train'):
        os.remove(os.path.join('./uploads_train', f))
    try:
        with open("./uploads_train/audio.mp3", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print('Audio is uploaded')

        data_own = loadAudioFolder('./uploads_train')
        addData2Mean(data_own,username)
        
    except:
        print("Error in uploading file")
        return {"status": 'NotOK'}
    return {'status':'OK', "username": username}

def voice_to_text_FPT(audio_url):
    url = 'https://api.fpt.ai/hmi/asr/general'
    payload = open(audio_url, 'rb').read()
    headers = {
        'api-key': 'ZWA4PFzx1ClIR8F5Cr0DMII7ONdnUH0r'
    }
    response = requests.post(url=url, data=payload, headers=headers)
    return response.json()['hypotheses'][0]['utterance']

def get_info_from_json(user_name, intent_class):
    # Opening JSON file
    f = open('data.json', encoding="utf8")
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    answer_from_intent = data[user_name][intent_class]
    return answer_from_intent

@app.post("/predict")
async def predictAIO(file: UploadFile = File(...)):
    try:
        with open("./uploads_train/audio-predict.mp3", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print('Audio is uploaded to predict')

        url_audio = "./uploads_train/audio-predict.mp3"

        #call function Speech2Text 
        sentences = voice_to_text_FPT(url_audio)
        print("sentences", sentences, "type:", type(sentences))
        classes = predict_intent_tfidf(sentences,tfidf_pickle, model)
        print("Intent predicted, class", classes)


        ###########PROCESS VOICE IDENTIFICATION##################
        data_x_vector = extract_Xvector(url_audio)
        print('input data shape', data_x_vector.shape)
        print('Predicting....')
        label_predict = predictVoice(data_x_vector.cpu().detach().numpy())
        print("Hi "+ label_predict)

        user = label_predict
        action = get_info_from_json(label_predict, classes)
    except:
        return {"status": "notOK"}
    return {"status": "OK", "action":action, "user":user}
