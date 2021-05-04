from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import urllib
import PIL.Image as Image
import cv2
import torch
import torchvision
from IPython.display import display
from sklearn.model_selection import train_test_split
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import os
import glob

import detect as dt
import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
import numpy as np

import io
from starlette.responses import StreamingResponse

from serve_model import predict, read_imagefile
#from application.schema import Symptom
#from application.components.prediction import symptom_check

app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>
<h2>This app is under development stage</h2>
<br>Initially it is trained on 10 categories of puffed food"""

app = FastAPI(title='Retail Product Checkout', description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "PNG")
    if not extension:
        return "Image must be jpg or png format!"
    print(file.filename)
    image = read_imagefile(await file.read())
    image.save('inference/images/'+file.filename)
    print('Saved as ','rpc/images/'+file.filename)
    #prediction = predict(image)
    #a = np.asarray(image)
    #print(imptest.add(2,1))
    #print(file.filename)
    a= dt.pre_detect(out='inference/output',source='inference/images')
    
    files = glob.glob('inference/images/*.jpg')

    if files != []:
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

    #return StreamingResponse(io.BytesIO(image.tobytes()), media_type="image/jpg")  
    return {'resp': a} #prediction


'''@app.post("/api/covid-symptom-check")
def check_risk(symptom: Symptom):
    return symptom_check.get_risk_level(symptom)
'''

'''@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    #prediction = predict(image)

    return prediction

@app.post("/api/covid-symptom-check")
def check_risk(symptom: Symptom):
    return symptom_check.get_risk_level(symptom)'''


if __name__ == "__main__":
    uvicorn.run(app, debug=True)