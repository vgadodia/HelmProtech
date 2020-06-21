# I'm trying to use this: https://flask-pymongo.readthedocs.io/en/latest/

import pandas as pd
from flask import Flask, jsonify, request, send_file, render_template, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import pathlib
import hashlib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def getupload():
    if request.method == "POST":
        try:
            memory = request.files['memory']
            description = request.form['description']
            if memory.filename == "" and description == "":
                return render_template('upload.html', errorMessage="Please either upload a photo or link a url")
            elif memory.filename != "" and description != "":
                return render_template('upload.html', errorMessage="Please either upload a photo or link a url.")
            else:
                print(memory.filename)
        except:
            return render_template('upload.html', errorMessage="Please either upload a photo or link a url.")
    return redirect("/Results")

@app.route('/Results')
def Results():
    # for memory in image:
    #     if len(memory["new sentences"]) == 0:
    #         return render_template('Results.html', errorMessage="", image=url_for('file', filename=memory["file"]))
    return render_template('Results.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)
