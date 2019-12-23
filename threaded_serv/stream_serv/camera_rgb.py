import os
import cv2
from base_camera import BaseCamera
import pyrealsense2 as rs
import numpy as np


class CameraRGB(BaseCamera):

    def __init__(self, pipeline):
        CameraRGB.pipeline = pipeline
        super(CameraRGB, self).__init__(pipeline)

    @staticmethod
    def frames():

        while True:
            frames = CameraRGB.pipeline.wait_for_frames()
            rgb = frames.get_color_frame()
            n_rgb = np.asanyarray(rgb.get_data())

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', n_rgb)[1].tobytes()
