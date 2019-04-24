import os
import time
import numpy as np
import cv2
# from matplotlib import pyplot as plt
from rice_detection import avgColor, cropBankNote, colorCorrection
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory

__author__ = 'Imbalance'

# app = Flask(__name__)
app = Flask(__name__, static_folder="static")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, 'static/')

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    upload = request.files.getlist("file")[0]
    print(upload)
    print("{} is the file name".format(upload.filename))
    filename = upload.filename
    destination = "/".join([target, filename])
    print ("Accept incoming file:", filename)
    print ("Save it to:", destination)
    upload.save(destination)

    read_img(destination)

    # return send_from_directory("static", filename)
    return render_template("complete.html", image_name=filename)

@app.route('/upload/<filename>', methods=["GET"])
def send_image(filename):
    return send_from_directory("static", filename)

def read_img(path):
    print('Read image at', path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred_rice(img)
    colorCorrection(img)

def pred_rice(img):
    print(img.shape)


if __name__ == "__main__":
    app.run(port=4555, debug=True)