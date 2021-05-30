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
from datetime import date
import shutil

import os
import glob
import sys
from azure.storage.blob import BlockBlobService, PublicAccess

import detect as dt
import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
import numpy as np
import pandas as pd

import io
#from starlette.responses import StreamingResponse

from serve_model import predict, read_imagefile
#from application.schema import Symptom
#from application.components.prediction import symptom_check

app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>
<h2>This app is under development stage</h2>
<br>Initially it is trained on 10 categories of puffed food"""

app = FastAPI(title='Retail Product Checkout', description=app_desc)

blob_service_client = BlockBlobService(
            account_name='23237iba', account_key='0jdjVZ68D9aVSZqcVJOkrvV3X8+/kUZU/Lza3GKqzu4vhOdCuCLkg3SgV8DxEbymI/PrEFwkLXatr2RUg+vxrw==')
container_name = 'dspd-price-container'


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
    user_price = pd.read_csv('price/price.csv')
    a = dt.pre_detect(out='inference/output',source='inference/images')
    bill = {}
    if len(a)!=0:
        for item in a:
            try:
                bill[item] = int(user_price.loc[user_price['Product_name']==item,['Price']].values)*int(a[item])
            except:
                bill[item] = 'price not found'
    
    files = glob.glob('inference/images/*.jpg')

    if files != []:
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

    #return StreamingResponse(io.BytesIO(image.tobytes()), media_type="image/jpg")  
    return {'qty': a,'bill':bill} #prediction

@app.post("/update_price")
async def get_csv(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("csv")
    if not extension:
        return "File must be csv format!"
    if not os.path.exists('price'): os.makedirs('price')
    lf = open("price/log.txt", "a")
    today = date.today()
    lf.write(f"Price update on {today} ")
    lf.close()
    print(type(file))
    with open('price/price.csv', 'wb') as f:
        shutil.copyfileobj(file.file, f)
    user_price = pd.read_csv('price/price.csv')
    try:
        blob_service_client.delete_blob(container_name, 'price.csv',delete_snapshots='include')
    except:
        pass
    folder_name = 'price'
    full_path_to_file = os.path.join(os.getcwd(),folder_name)
    full_path_to_file = os.path.join(full_path_to_file,'price.csv')
    
    blob_service_client.create_blob_from_path(
            container_name, 'price.csv', full_path_to_file)
    
    #category.
    #for ind,row in user_price.iterrows():
    #    print(row)
    #user_price.loc[user_price['Product_name'] == "1_puffed_food'",['Price']]=5
    #print(user_price.loc[user_price['Product_name'] == "1_puffed_food'"]['Price'])
    return {'no. of categories updated succesfully': len(user_price), 'next_step': 'run azure train script manually'}


if __name__ == "__main__":
    uvicorn.run(app, debug=True)