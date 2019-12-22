#from flask import Flask

#app = Flask(__name__, static_folder='.', static_url_path='')

# @app.route('/')
# def index():
# return app.send_static_file#('index.html')
# @app.route('/hello/<name>')
# def hello(name):
# return name

#app.run(port=8000, debug=True)


import os
import io
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
import matplotlib.pyplot as plt
import math
import glob
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMAGE_WIDTH = 640
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']
        #number = int(request.form['number'])
        # 変なファイル弾き
        if img_file and allowed_file(img_file.filename) and request.form['number']:
            filename = secure_filename(img_file.filename)
        else:
            return ''' <p>人数と画像選んでね</p> '''
        # BytesIOで読み込んでOpenCVで扱える型にする
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # とりあえずサイズは小さくする
        raw_img = cv2.resize(img, (IMAGE_WIDTH, int(
            IMAGE_WIDTH*img.shape[0]/img.shape[1])))
        # サイズだけ変えたものも保存する
        #raw_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_'+filename)
        #cv2.imwrite(raw_img_url, raw_img)
        # なにがしかの加工
        #import numpy as np
        #import cv2
        #import matplotlib.pyplot as plt
        img = raw_img
        canny_img = cv2.Canny(img, 230, 500)
        # find hough circles
        circles = cv2.HoughCircles(canny_img, cv2.HOUGH_GRADIENT,
                                   dp=2, minDist=100, param1=25, param2=30,
                                   minRadius=200, maxRadius=210)
        print(circles)
        number = int(request.form['number'])
        k = number - 1
        cups_circles = np.copy(img)
        # if circles are detected, draw them
        if circles is not None and len(circles) > 0:
            # note: cv2.HoughCircles returns circles nested in an array.
            # the OpenCV documentation does not explain this return value format
            circles = circles[0]
            for (x, y, r) in circles:
                x, y, r = int(x), int(y), int(r)
                cv2.circle(cups_circles, (x, y), r, (255, 255, 0), 4)
                for i in range(k+1):
                    cv2.line(cups_circles, (x, y), (x+int(r*math.cos(math.radians((360/(k+1))*(i)))), y+int(
                        r*math.sin(math.radians((360/(k+1))*(i))))), (255, 0, 0), thickness=5, lineType=cv2.LINE_8, shift=0)
            #gray_img=cv2.cvtColor(cups_circles, cv2.COLOR_BGR2RGB)
            gray_img = cups_circles
        #既にある画像の削除
        #if os.path.join(app.config['UPLOAD_FOLDER'], 'gray_'+filename):
            #file_list = glob.glob('*.jpg')
            #for file in file_list:
                #print("remove：{0}".format(file))
                #os.remove(file)
        # 加工したものを保存する
        gray_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'gray_'+filename)

        cv2.imwrite(gray_img_url, gray_img)

        # return render_template('index.html', raw_img_url=raw_img_url, gray_img_url=gray_img_url)
        return render_template('index.html', gray_img_url=gray_img_url)
        # return render_template('index.html', gray_img_url=gray_img)

    else:
        return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.debug = True
    app.run()
