#!/usr/bin/python3

import pyrealsense2 as rs
import numpy as np
import cv2
import scipy.misc

#### server
import socket
import time
from imutils.video import VideoStream
from imagezmq import ImageSender


class RealsenseCam():

    def __init__(self):
        w = 640
        h = 480
        fps = 30
        resolution = (w, h)
        
        # configuration pipeline
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()



        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.pc = rs.pointcloud()
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 2**1)

        ### fixed to two ports rgb and depth port
        # the image tcp port
        self.rgb_send = ImageSender(connect_to='tcp://127.0.0.1:5555', REQ_REP=True)
        # the depth tcp port
        self.depth_send = ImageSender(connect_to='tcp://127.0.0.1:5556', REQ_REP=True)
        self.cam_name = socket.gethostname()
        self._start_pipeline()
        time.sleep(2)


    def _start_pipeline(self):

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        #depth_image = np.asanyarray(aligned_depth_frame.get_data())
        self.color_image = np.asanyarray(color_frame.get_data())

        #####
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        points = self.pc.calculate(depth_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        self.verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        self.texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv
        self.cam_name = 'realsensecam'


    # outputs the raw color image data
    def get_color_data(self):
        self.rgb_send.send_image(self.cam_name, self.color_image)

    # outputs the raw depth image data
    def get_depth_data(self):
        self.depth_send.send_image(self.cam_name, self.verts)


if __name__ == '__main__':
    cam = RealsenseCam()

    while True:

        depth_image= cam.get_depth_data()
        color_image =cam.get_color_data()


        # Render images
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', color_image)
        key = cv2.waitKey(1) 









