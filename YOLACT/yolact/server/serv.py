import time
import sys
import os
import subprocess
import json
import cv2
from flask import Flask, jsonify, request, g, session, stream_with_context, Response, render_template
#from torch.multiprocessing import Pool, Process, Value, set_start_method
from multiprocessing import Process, Value

# import eval
sys.path.append('/home/ld/04_MIT/6881_Intelligent_Robot_M/FINAL_PROJECT/YOLACT/yolact')
#import eval as ev
from camera import RealsenseCam as cam
import pcl_server_mult_client as mc

app = Flask(__name__)

####STARTYOLACT#####
@app.route('/api/yolact', methods=['GET','POST'])
def yolact():
    #ev.main()
    return 'start yolact...'

@app.route('/api/cam_feed_start', methods=['GET','POST'])
def cam_feed_start():
    mc()
    #while True:
    #    cams.get_color_data()
    #    cams.get_depth_data()
    #    time.sleep(0.01)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)

