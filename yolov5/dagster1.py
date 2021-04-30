# -*- coding: utf-8 -*-
"""DAGster.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hWY8pBjbra-2jY-mcgh0XO4KUdY9iXeJ
"""

#pip install dagster

# Commented out IPython magic to ensure Python compatibility.
from dagster import execute_pipeline, pipeline, solid
#import main

@solid
def get_name(_):
    return 'dagster'

@solid
def hello(context):
    context.log.info(main.py)
#     %cd yolov5
#    !python train.py --img 640 --batch 4 --epochs 30 \--data ./data/rpc.yaml --cfg ./models/yolov5l.yaml --weights yolov5x.pt \  --name yolov5x_rpc --cache
#    !python detect.py --weights weights/best_yolov5x_rpc.pt \ --img 640 --conf 0.4 --source ./inference/images/
    #!python main.py
    
@pipeline
def hello_pipeline():
    # hello(get_name())
    hello()

#if __name__ == "__main__":
#    result = execute_pipeline(hello_pipeline)