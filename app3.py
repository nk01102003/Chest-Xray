
from flask import Flask, render_template, flash, request, session,send_file
from flask import render_template, redirect, url_for, request
import warnings
import datetime
import cv2


app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route("/")
def homepage():

    return render_template('index.html')



@app.route("/Training")
def Training():
    return render_template('Tranning.html')

@app.route("/Test")
def Test():
    return render_template('Test.html')




@app.route("/train", methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        import model as model

        return render_template('Tranning.html')





@app.route("/testimage", methods=['GET', 'POST'])
def testimage():
    if request.method == 'POST':


        file = request.files['fileupload']
        file.save('static/Out/Test.jpg')

        img = cv2.imread('static/Out/Test.jpg')
        if img is None:
            print('no data')

        img1 = cv2.imread('static/Out/Test.jpg')
        print(img.shape)
        img = cv2.resize(img, ((int)(img.shape[1] / 5), (int)(img.shape[0] / 5)))
        original = img.copy()
        neworiginal = img.copy()
        cv2.imshow('original', img1)
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        img1S = cv2.resize(img1, (960, 540))

        cv2.imshow('Original image', img1S)
        grayS = cv2.resize(gray, (960, 540))
        cv2.imshow('Gray image', grayS)

        gry = 'static/Out/gry.jpg'

        cv2.imwrite(gry, grayS)
        from PIL import  ImageOps,Image

        im = Image.open(file)

        im_invert = ImageOps.invert(im)
        inv = 'static/Out/inv.jpg'
        im_invert.save(inv, quality=95)

        dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
        cv2.imshow("Nosie Removal", dst)
        noi = 'static/Out/noi.jpg'

        cv2.imwrite(noi, dst)

        import warnings
        warnings.filterwarnings('ignore')

        import tensorflow as tf
        classifierLoad = tf.keras.models.load_model('model.h5')

        import numpy as np
        from keras.preprocessing import image

        test_image = Image.open('static/Out/Test.jpg')
        test_image = test_image.resize((224, 224))  # Resize the image to match the model's input shape
        # test_image = ImageOps.invert(test_image)  # Invert the image
        # test_image = test_image.convert('RGB')
        test_image = np.array(test_image)  # Convert to NumPy array
        test_image = test_image / 255.0  # Normalize the pixel values (if necessary)
        test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

        result = classifierLoad.predict(test_image)
        print(result)
        out = ''
        fer=''
        predicted_class = np.argmax(result)

        # Define the classes
        classes = ["COVID19","NORMAL", "PNEUMONIA"]

        # Check the predicted class
        if predicted_class == 0:
            out = classes[0]
        elif predicted_class == 1:
            out = classes[1]
        elif predicted_class == 2:
            out = classes[2]
        else:
            out= "NOT PREDICT"


        print(out)
        org = 'static/Out/Test.jpg'
        gry ='static/Out/gry.jpg'
        inv = 'static/Out/inv.jpg'
        noi = 'static/Out/noi.jpg'




        return render_template('Test.html',fer=1,result=out,org=org,gry=gry,inv=inv,noi=noi)




if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
