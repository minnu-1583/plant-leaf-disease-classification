from __future__ import division, print_function
import sys
import os
import numpy as np
import sqlite3
import cv2
import pathlib

from keras.models import load_model
from ultralytics import YOLO

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Fix Windows Path Issue
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

# Store current disease for graph
current_disease = "No Data"

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'
app.config['MODEL_PATH1'] = 'detection/data1/best.pt'
app.config['MODEL_PATH2'] = 'detection/data2/best.pt'
app.config['MODEL_PATH3'] = 'detection/data3/best.pt'


# ================= Detection Models =================

model1 = YOLO(app.config['MODEL_PATH1'])
model2 = YOLO(app.config['MODEL_PATH2'])
model3 = YOLO(app.config['MODEL_PATH3'])

UPLOAD_FOLDER = 'static/uploads/'


# ================= Classification Models =================

CTS1 = load_model('xcep_data1.h5', compile=False)
CTS2 = load_model('xcep_data2.h5', compile=False)
CTS3 = load_model('xcep_data3.h5', compile=False)


# ================= Classification Functions =================

def model_predict1(image_path, model):

    image = load_img(image_path, target_size=(128,128))
    image = img_to_array(image)/255
    image = np.expand_dims(image, axis=0)

    result = np.argmax(model.predict(image))

    classes = [
        "APPLE BLACK ROT",
        "APPLE SCAB",
        "CORN GRAY LEAF SPOT",
        "CORN NORTHERN LEAF BLIGHT",
        "POTATO EARLY BLIGHT",
        "POTATO LATE BLIGHT"
    ]

    return classes[result]


def model_predict2(image_path, model):

    image = load_img(image_path, target_size=(128,128))
    image = img_to_array(image)/255
    image = np.expand_dims(image, axis=0)

    result = np.argmax(model.predict(image))

    classes = [
        "GRAPE BLACK MEASLES",
        "GRAPE BLACK ROT",
        "GRAPE HEALTHY",
        "GRAPE LEAF SPOT"
    ]

    return classes[result]


def model_predict3(image_path, model):

    image = load_img(image_path, target_size=(128,128))
    image = img_to_array(image)/255
    image = np.expand_dims(image, axis=0)

    result = np.argmax(model.predict(image))

    classes = [
        "TOMATO EARLY BLIGHT",
        "TOMATO HEALTHY",
        "TOMATO LATE BLIGHT",
        "TOMATO MOSAIC VIRUS",
        "TOMATO SEPTORIA LEAF SPOT",
        "TOMATO YELLOW LEAF CURL VIRUS"
    ]

    return classes[result]


# ================= Routes =================

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/logon')
def logon():
    return render_template('signup.html')


@app.route('/login')
def login():
    return render_template('signin.html')


# ================= Signup =================

@app.route("/signup", methods=['GET','POST'])
def signup():

    if request.method == 'POST':

        username = request.form.get('user')
        name = request.form.get('name')
        email = request.form.get('email')
        number = request.form.get('mobile')
        password = request.form.get('password')

        con = sqlite3.connect('signup.db')
        cur = con.cursor()

        # Check if username already exists
        cur.execute("SELECT * FROM info WHERE user=?", (username,))
        existing_user = cur.fetchone()

        if existing_user:
            return render_template("signup.html",
                                   error="Username already exists")

        # Insert new user
        cur.execute(
            "INSERT INTO info (user,email,password,mobile,name) VALUES (?,?,?,?,?)",
            (username, email, password, number, name)
        )

        con.commit()
        con.close()

        return render_template("signin.html",
                               success="Account created successfully. Please login")

    return render_template("signup.html")


# ================= Login =================

@app.route("/signin", methods=['GET','POST'])
def signin():

    if request.method == 'POST':

        user = request.form.get('user')
        password = request.form.get('password')

        con = sqlite3.connect('signup.db')
        cur = con.cursor()

        cur.execute(
            "SELECT user,password FROM info WHERE user=? AND password=?",
            (user,password)
        )

        data = cur.fetchone()
        con.close()

        if data:
            return redirect(url_for('index1'))
        else:
            return render_template("signin.html",
                                   error="Invalid Credentials")

    return render_template("signin.html")


# ================= Dashboard =================

@app.route('/index1')
def index1():
    return render_template('index1.html')


@app.route('/index4')
def index4():
    return render_template('index4.html')


# ================= Graph Page =================

@app.route('/about1')
def about1():
    global current_disease
    return render_template('about1.html', disease=current_disease)


# ================= Classification =================

@app.route('/predict1', methods=['POST'])
def predict1():

    global current_disease

    file = request.files['file']
    filename = file.filename

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    disease = model_predict1(file_path, CTS1)
    current_disease = disease

    return render_template("result.html",
                           pred_output=disease,
                           img_src=UPLOAD_FOLDER + filename,
                           disease=disease)


@app.route('/predict2', methods=['POST'])
def predict2():

    global current_disease

    file = request.files['file']
    filename = file.filename

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    disease = model_predict2(file_path, CTS2)
    current_disease = disease

    return render_template("result.html",
                           pred_output=disease,
                           img_src=UPLOAD_FOLDER + filename,
                           disease=disease)


@app.route('/predict3', methods=['POST'])
def predict3():

    global current_disease

    file = request.files['file']
    filename = file.filename

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    disease = model_predict3(file_path, CTS3)
    current_disease = disease

    return render_template("result.html",
                           pred_output=disease,
                           img_src=UPLOAD_FOLDER + filename,
                           disease=disease)


# ================= Detection =================

def process_image(image_path, output_path, model):

    img = cv2.imread(image_path)
    results = model.predict(source=img)

    for result in results:
        for box in result.boxes:

            x1,y1,x2,y2 = box.xyxy[0].int().tolist()
            label = model.names[int(box.cls[0])]

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    cv2.imwrite(output_path,img)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'output/{filename}'))


@app.route('/predict4', methods=['POST'])
def predict4():

    file = request.files['file']
    filename = secure_filename(file.filename)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)

    process_image(file_path, output_path, model1)

    return redirect(url_for('uploaded_file', filename=filename))


if __name__ == '__main__':
    app.run(debug=True)