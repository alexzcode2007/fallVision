import requests
import json
import base64
import random
import cv2

BASE = "http://34.211.13.26:5000/"

def gen_name():
    numbers = [random.randint(65, 65+26*2-1) for i in range(10)]
    numbers = list(map(lambda x: x+6 if x > 90 else x, numbers))
    numbers = list(map(chr, numbers))
    return ''.join(numbers)
    

def save_file(file):
    data = {"name": gen_name() + '.jpg', "file": base64.b64encode(cv2.imencode('.jpg', file)[1]).decode("utf-8")}
    
    headers = {"Content-type" : "application/json"}
    
    response = requests.post(BASE, data=json.dumps(data),  headers=headers)
    
    with open("response_log.txt", "a") as log:
        log.write(response.text)
