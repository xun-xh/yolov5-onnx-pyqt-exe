import json
import os

import cv2
import flask
from datetime import timedelta

from utils import detect

app = flask.Flask(__name__,
                  static_folder='httpserver/out',
                  template_folder='httpserver/templates')

app.secret_key = '666666'
app.config['UPLOAD_FOLDER'] = 'httpserver/data'  # 上传的文件保存目录
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=5)
print(app.config['SEND_FILE_MAX_AGE_DEFAULT'])

model = detect.YOLOv5()  # 导入模型，以便节省detect时间
model.initModel('need/models/yolov7-tiny.onnx')
model.initConfig(class_names=open(
    'need/yolov7-tiny.txt', 'r').read().split('\n'))


@app.route('/')
def index():
    return flask.redirect('online')


@app.route('/online', methods=['GET', 'POST'])
def online():
    """可视化目标检测"""
    if flask.request.method == 'POST':
        if 'file' not in flask.request.files:
            flask.flash('not file part!')
            return flask.redirect(flask.request.url)

        f = flask.request.files['file']
        if f.filename == '':
            flask.flash('not file upload')
            return flask.redirect(flask.request.url)

        if f and f.filename.endswith(detect.DataLoader.IMAGE_TYPE):
            # filename = secure_filename(f.filename)
            # secure_filename 不支持中文文件名称的获取……

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')
            f.save(filepath)

            dataset = detect.DataLoader(filepath)
            img, _ = next(dataset)
            model.detect(img)
            return_img_path = 'httpserver/out/test.jpg'
            cv2.imwrite(return_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            return flask.render_template('online_ok.html')
    return flask.render_template('online.html')


@app.route('/api', methods=['GET', 'POST'])
def api():
    """api接口, 接受图片"""
    if flask.request.method == 'POST':
        if 'file' not in flask.request.files or not flask.request.files['file'].filename:
            return {'code': 1, 'msg': '请上传有效文件'}

        f = flask.request.files['file']
        if not f or not f.filename.endswith(detect.DataLoader.IMAGE_TYPE):
            return {'code': 1, 'msg': '不支持的文件格式'}

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')
        f.save(filepath)

        img = cv2.imread(filepath)
        model.conf_threshold = flask.request.args.get('conf', 0.5, float)
        model.iou_threshold = flask.request.args.get('iou', 0.5, float)
        model.with_pos = flask.request.args.get('pos', True, bool)
        res = model.detect(img)
        return json.dumps({'code': 0, 'msg': str(res)})
    elif flask.request.method == 'GET':
        return {'code': 1, 'msg': '只支持POST'}


class CheckOnline(object):
    def __init__(self):
        self.dataset = detect.DataLoader(0)

    @staticmethod
    @app.route('/check')
    def check():
        """远程查看"""
        return flask.render_template('check.html')

    @staticmethod
    @app.route('/video_feed')  # 这个地址返回视频流响应
    def video_feed():
        return flask.Response(CheckOnline().gen(),
                              mimetype='multipart/x-mixed-replace; boundary=frame')

    def detect(self):
        img, _ = next(self.dataset)
        model.detect(img)
        ret, jpeg = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return jpeg.tobytes()

    def gen(self):
        while True:
            frame = self.detect()
            # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=2222)
