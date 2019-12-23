import os
import cv2
from base_camera import BaseCamera
import pyrealsense2 as rs
import numpy as np
import math


class Camera(BaseCamera):

    def __init__(self, pipeline):
        Camera.pipeline = pipeline
        super(Camera, self).__init__(pipeline)

    @staticmethod
    def frames_rgb():
        try:
            while True:
                frames_rgb = Camera.pipeline.wait_for_frames()
                rgb = frames_rgb.get_color_frame()
                n_rgb = np.asanyarray(rgb.get_data())

                # encode as a jpeg image and return it
                yield cv2.imencode('.jpg', n_rgb)[1].tobytes()
        finally:
            Camera.pipeline.stop()


    @staticmethod
    def frames_dpt():
        align_to = rs.stream.color
        align =rs.align(align_to)

        try:
            pc = rs.pointcloud()
            decimate = rs.decimation_filter()
            decimate.set_option(rs.option.filter_magnitude, 2 ** 1)

            while True:

                f = Camera.pipeline.wait_for_frames()
                fr_a = align.process(f)
                frames = fr_a.get_depth_frame()

                # 240, 320
                #dpt = frames.get_depth_frame()
                dpt = decimate.process(frames)

                points = pc.calculate(dpt)
                v = points.get_vertices()

                # 76800,3
                verts = np.asanyarray(v).view(np.float32).reshape(-1,3)

                start_byte = b'\xff\xd8\xf1'
                end_byte = b'\xff\xd9\xf2' 

                fin_b = verts.tobytes()
                data_ = start_byte + fin_b + end_byte


                yield data_

        finally:
            Camera.pipeline.stop()




class ReduceVerts:

    def __init__(self):
        self.pitch, self.yaw = math.radians(-10), math.radians(-90)
        self.translation = np.asarray([0,0, -1], dtype=np.float32)
        self.distance = 2


    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0,0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

    def view(self,v):
        return np.dot(v-self.pivot, self.rotation) + self.pivot - self.translation

    # returns the calc verts 
    def calc_verts(self, v):
        return np.asanyarray(v).view(np.float32).reshape(-1,3)

