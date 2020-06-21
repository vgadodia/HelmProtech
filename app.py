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

warnings.filterwarnings("ignore")

app = Flask(__name__)

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
                image, mot, hel = predict(memory.filename)
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
    # for memory in image:
    #     if len(memory["new sentences"]) == 0:
    #         return render_template('Results.html', errorMessage="", image=url_for('file', filename=memory["file"]))
    
    # image = image.decode("utf-8")
    # image = image.decode("utf-8")
    print(image)
    try:
        thing = helmet/motor*100
    except: thing = 0
    # retval, buffer = cv.imencode('.png', image)
    # response = make_response(buffer.tobytes())
    # return response
    # try:
    return render_template('results.html', memory_vs_time_rf = url_for("static", filename="new_output.jpg"), perfect_rate=thing, forget_rate=motor-helmet)
    # except:
    #     return render_template('results.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
