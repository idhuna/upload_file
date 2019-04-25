import os
import time
import numpy as np
import cv2
# from matplotlib import pyplot as plt
from rice_detection import rice_detection
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory

__author__ = 'Imbalance'

# app = Flask(__name__)
app = Flask(__name__, static_folder="static")
app.config.from_object(os.environ['APP_SETTINGS'])

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, 'static/')

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    result = request.form
    crop_detail = [int(e) for e in result['info'].split(',')]
    print(crop_detail)
    target = os.path.join(APP_ROOT, 'static')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    upload = request.files.getlist("file")[0]
    # print(upload.read()[:100])
    print("{} is the file name".format(upload.filename))
    filename = upload.filename
    destination = "/".join([target, filename])
    print ("Accept incoming file:", filename)
    print ("Save it to:", destination)
    upload.save(destination)

    test_rice_name = rice_detection(destination, filename, crop_detail)

    # return send_from_directory("static", filename)
    return render_template("complete.html", image_name=test_rice_name)

@app.route('/upload/<filename>', methods=["GET"])
def send_image(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(port=4555, debug=True)