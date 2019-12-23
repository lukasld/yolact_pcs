#!/usr/bin/env python
import os
from importlib import import_module
import pyrealsense2 as rs
from flask import Flask, render_template, Response, stream_with_context, request, send_file, make_response
import json
import cv2
import numpy as np

# import camera driver
from camera_dpt_rgb import Camera

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

pip = Camera(pipeline)
app = Flask(__name__)

box_val = None

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen_rgb(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame_rgb()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_dpt(camera):
    """Video streaming generator function."""
    while True:
        byte = camera.get_frame_dpt()
        yield (b'--byte\r\n'
               b'Content-Type: byte\r\n\r\n' + byte + b'\r\n')



@app.route('/video_feed_rgb')
def video_feed_rgb():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(stream_with_context(gen_rgb(pip)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_dpt')
def video_feed_dpt():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_dpt(pip),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# receives the mask
@app.route('/mask', methods = ['GET','POST'])
def mask():
    if request.method == 'POST':
        r = request
        arr = np.fromstring(r.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        cv2.imwrite('tmp_mask.png', img)
        return Response(status=200, mimetype='application/json')

    if request.method == 'GET':
        return send_file('tmp_mask.png', mimetype='image/png')

# receives the box
@app.route('/box', methods = ['GET','POST'])
def box():
    if request.method == 'POST':
        r = request.get_json()
        with open('box_dim.json', 'w') as f:
            json.dump(r, f)
        return Response(status=200, mimetype='application/json')

    if request.method == 'GET':
        data = None
        with open('box_dim.json', 'r') as f:
            data = f.read()
        return data


if __name__ == '__main__':
    # pretty unsecure but for test purposes ok
    app.run(host='10.150.0.37', threaded=True)

    

