# I'm trying to use this: https://flask-pymongo.readthedocs.io/en/latest/

import pandas as pd
from flask import Flask, jsonify, request, send_file, render_template, redirect, url_for, make_response
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import pathlib
import hashlib
from yolo import predict
import cv2 as cv
import warnings

import random

kk = str(random.randint(1, 100000000))

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

image = ""
motor = 0
helmet = 0

@app.route('/')
def index():
    # print(predict('input1.jpg'))
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def getupload():
    global image, motor, helmet
    if request.method == "POST":
        # memory = request.files['memory']
        # image, mot, hel = predict(memory.filename)
        # motor+=mot
        # helmet+=hel
        # return redirect('/upload')
        try:
            memory = request.files['memory']
            description = request.form['description']
            print(memory.filename == "", description == "")
            if memory.filename != "" and description != "":
                print("One")
                return render_template('upload.html', errorMessage="Please either upload a photo or link a url")
            # elif memory.filename != "" and description != "":
            #     return render_template('upload.html', errorMessage="Please either upload a photo or link a url.")
            elif memory.filename != "":
                print("Two")
                print(memory.filename)
                print("here1")
                image, mot, hel = predict(memory.filename)
                cv.imwrite("/static/new_output" + kk +  ".jpg", image)
                print("here2")
                motor+=mot
                helmet+=hel
                print(memory.filename)
                print("END")
            elif description != "":
                print("Three")
                print(description)
                image, stats = predict(description)
                motor+=stats[0]
                helmet+=stats[1]
                print(memory.filename)
            else:
                print("Four")
                return render_template('upload.html', errorMessage="Please either upload a photo or link a url")
        except:
            print("EXCEPT")
            return render_template('upload.html', errorMessage="Please either upload a photo or link a url.")
    return redirect("/results")

@app.route('/results')
def results():
    global image, motor, helmet
    
    print(image)

    try:
        thing = helmet/motor*100
    except: thing = 0
    
    return render_template('results.html', memory_vs_time_rf = "/static/new_output" + kk + ".jpg", perfect_rate=thing, forget_rate=motor-helmet)
    

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
