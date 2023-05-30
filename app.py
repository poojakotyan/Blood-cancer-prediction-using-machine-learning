from flask import Flask,render_template,redirect,request,send_from_directory
import tensorflow as tf
#import matplotlib.pyplot as plt;
from keras import layers;
import os
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import keras.utils as image

model_file = "model/model.h5"

model = tf.keras.models.load_model(model_file)

app = Flask(__name__,template_folder='templates')
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

class_names=np.array(['Cancer', 'Normal'])

def ImagePrediction(loc):
    test_image = image.load_img(loc, target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis =0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        predictions = 'Normal'
    else:
        predictions = 'Cancer'
    return predictions

@app.route('/home.html',methods=['GET','POST'])
def home():
    if request.method=='POST':
        if 'img' not in request.files:
            return render_template('home.html',filename="unnamed.png",message="Please upload an file")
        f = request.files['img'] 
        filename1 = secure_filename(f.filename) 
        if f.filename=='':
            return render_template('home.html',filename="unnamed.png",message="No file selected")
        if not ('jpeg' in f.filename or 'png' in f.filename or 'jpg' in f.filename):
            return render_template('home.html',filename="unnamed.png",message="please upload an image with .png or .jpg/.jpeg extension")
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if len(files)==1:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename1))
        else:
            files.remove("unnamed.png")
            file_ = files[0]
            os.remove(app.config['UPLOAD_FOLDER']+'/'+file_)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename1))
        predictions = ImagePrediction(os.path.join(app.config['UPLOAD_FOLDER'],filename1))
        return render_template('home.html',filename=filename1,message=predictions,show=True)
    return render_template('home.html',filename='unnamed.png')

@app.route('/profile.html',methods=['GET','POST'])
def profile():
    return render_template('profile.html')

@app.route('/info.html', methods=['GET', 'POST'])
def info():
    return render_template('info.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)