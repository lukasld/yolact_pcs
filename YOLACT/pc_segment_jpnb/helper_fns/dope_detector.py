### Parts of this code are from NVIDIA's code release. See license below. ###

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from __future__ import print_function

import os
import cv2
import yaml
import time
import numpy as np
import copy

import transformations
from PIL import Image
from PIL import ImageDraw

from pydrake.math import RollPitchYaw, RotationMatrix

import torch

from dope.inference.cuboid import Cuboid3d
from dope.inference.cuboid_pnp_solver import CuboidPNPSolver
from dope.inference.detector import ObjectDetector, DopeNetwork

CONFIG_FILE = './config/config_pose.yaml'
with open(CONFIG_FILE, 'rb') as fid:
    CONFIG = yaml.load(fid)

class Pose(object):
    def __init__(self, pos, quat, toMeters=True):
        if toMeters:
            CONVERT_SCALE_CM_TO_METERS = 100
        else:
            CONVERT_SCALE_CM_TO_METERS = 1
        self._x = pos[0] / CONVERT_SCALE_CM_TO_METERS
        self._y = pos[1] / CONVERT_SCALE_CM_TO_METERS
        self._z = pos[2] / CONVERT_SCALE_CM_TO_METERS
        self.q = copy.deepcopy(quat)
    
    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z


class CameraConfig(object):
    def __init__(self):
        self.serial_no = "816612062437"
        #intrinsics
        self.width = 848
        self.height = 480
        self.focal_x = 635.491
        self.focal_y = 635.491
        self.center_x = 431.021
        self.center_y = 238.404
        #extrinsics
        self.x = -0.215
        self.y = -0.450
        self.z = 0.425
        self.roll = 2.839
        self.pitch = 1.337
        self.yaw = -2.997

    def P(self):
        return np.array([[self.focal_x, 0, self.center_x, 0],
                        [0, self.focal_y, self.center_y, 0],
						[0, 0, 1, 0]]).reshape((3,4))

    def T_cam2world(self):
        R = RollPitchYaw(self.roll, self.pitch, self.yaw).ToRotationMatrix().matrix()
        T = np.zeros((4,4), dtype=np.float64)
        T[0:3,0:3] = R
        T[0,3] = self.x
        T[1,3] = self.y
        T[2,3] = self.z
        T[3,3] = 1.0
        return T

class ModelData(object):
    '''This class contains methods for loading the neural network'''

    def __init__(self, name="", net_path="", gpu_id=0):
        self.name = name
        self.net_path = net_path  # Path to trained network model
        self.net = None  # Trained network
        self.gpu_id = gpu_id
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

    def get_net(self):
        '''Returns network'''
        if not self.net:
            self.load_net_model()
        return self.net

    def load_net_model(self):
        '''Loads network model from disk'''
        if not self.net and os.path.exists(self.net_path):
            self.net = self.load_net_model_path(self.net_path)
        if not os.path.exists(self.net_path):
            raise Exception("ERROR:  Unable to find model weights: '{}'".format(
                self.net_path))

    def load_net_model_path(self, path):
        '''Loads network model from disk with given path'''
        model_loading_start_time = time.time()
        print("Loading DOPE model '{}'...".format(path))
        net = DopeNetwork()
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            net = torch.nn.DataParallel(net, [0]).cuda()
        else:
            net = torch.nn.DataParallel(net, [0]).cpu()
            
        net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        net.eval()
        print('    Model loaded in {} seconds.'.format(
            time.time() - model_loading_start_time))
        return net

    def __str__(self):
        '''Converts to string'''
        return "{}: {}".format(self.name, self.net_path)



class Draw(object):
    """Drawing helper class to visualize the neural network output"""

    def __init__(self, im):
        """
        :param im: The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)

    def draw_line(self, point1, point2, line_color, line_width=2):
        """Draws line on image"""
        if point1 is not None and point2 is not None:
            self.draw.line([point1, point2], fill=line_color, width=line_width)

    def draw_dot(self, point, point_color, point_radius):
        """Draws dot (filled circle) on image"""
        if point is not None:
            xy = [
                point[0] - point_radius,
                point[1] - point_radius,
                point[0] + point_radius,
                point[1] + point_radius
            ]
            self.draw.ellipse(xy,
                              fill=point_color,
                              outline=point_color
                              )

    def draw_cube(self, points, color=(255, 0, 0)):
        """
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        """

        # draw front
        self.draw_line(points[0], points[1], color)
        self.draw_line(points[1], points[2], color)
        self.draw_line(points[3], points[2], color)
        self.draw_line(points[3], points[0], color)

        # draw back
        self.draw_line(points[4], points[5], color)
        self.draw_line(points[6], points[5], color)
        self.draw_line(points[6], points[7], color)
        self.draw_line(points[4], points[7], color)

        # draw sides
        self.draw_line(points[0], points[4], color)
        self.draw_line(points[7], points[3], color)
        self.draw_line(points[5], points[1], color)
        self.draw_line(points[2], points[6], color)

        # draw dots
        self.draw_dot(points[0], point_color=color, point_radius=4)
        self.draw_dot(points[1], point_color=color, point_radius=4)

        # draw x on the top
        self.draw_line(points[0], points[5], color)
        self.draw_line(points[1], points[4], color)


class DopeMain(object):
    def __init__(self):
        self.pubs = {}
        self.models = {}
        self.pnp_solvers = {}
        self.pub_dimension = {}
        self.draw_colors = {}
        self.dimensions = {}
        self.class_ids = {}
        self.model_transforms = {}
        self.meshes = {}
        self.mesh_scales = {}

        self.input_is_rectified = True
        self.downscale_height = 500

        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = 0.5
        self.config_detect.thresh_map = 0.01
        self.config_detect.sigma = 3
        self.config_detect.thresh_points = 0.1

        # For each object to detect, load network model, create PNP solver, and start ROS publishers
        model = 'cracker'
        self.models[model] = ModelData("cracker", "./resources/dope_weights/cracker_60.pth") 
        self.models[model].load_net_model()

        self.model_transforms[model] = np.array([0.0, 0.0, 0.0, 1.0], dtype='float64')
        self.mesh_scales[model] = 0.01

        self.draw_colors[model] = tuple(CONFIG["draw_colors"][model])
        self.dimensions[model] = tuple(CONFIG["dimensions"][model])
        self.class_ids[model] = CONFIG["class_ids"][model]

        self.pnp_solvers[model] = \
            CuboidPNPSolver(
                model,
                cuboid3d=Cuboid3d(CONFIG['dimensions'][model])
            )
        
        
    def image_callback(self, imgPath):
        """Image callback"""
        img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
        camera_info = CameraConfig()

        # Update camera matrix and distortion coefficients
        if self.input_is_rectified:
            P = np.matrix(camera_info.P(), dtype='float64')
            P.resize((3, 4))
            camera_matrix = P[:, :3]
            dist_coeffs = np.zeros((4, 1))
        else:
            camera_matrix = np.matrix(camera_info.K, dtype='float64')
            camera_matrix.resize((3, 3))
            dist_coeffs = np.matrix(camera_info.D, dtype='float64')
            dist_coeffs.resize((len(camera_info.D), 1))

        # Downscale image if necessary
        height, width, _ = img.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            img = cv2.resize(img, (int(scaling_factor * width), int(scaling_factor * height)))

        for m in self.models:
            self.pnp_solvers[m].set_camera_intrinsic_matrix(camera_matrix)
            self.pnp_solvers[m].set_dist_coeffs(dist_coeffs)

        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)
        
        for m in self.models:
            # Detect object
            results = ObjectDetector.detect_object_in_image(
                self.models[m].net,
                self.pnp_solvers[m],
                img,
                self.config_detect
            )
            # Publish pose and overlay cube on image
            for i_r, result in enumerate(results):
                if result["location"] is None:
                    continue
                loc = result["location"]
                ori = result["quaternion"]

                # transform orientation
                transformed_ori = transformations.quaternion_multiply(ori, self.model_transforms[m])

                # rotate bbox dimensions if necessary
                # (this only works properly if model_transform is in 90 degree angles)
                dims = rotate_vector(vector=self.dimensions[m], quaternion=self.model_transforms[m])
                dims = np.absolute(dims)
                dims = tuple(dims)

                """ 
                pose.position.x = loc[0] / CONVERT_SCALE_CM_TO_METERS
                pose.position.y = loc[1] / CONVERT_SCALE_CM_TO_METERS
                pose.position.z = loc[2] / CONVERT_SCALE_CM_TO_METERS
                pose.orientation.x = transformed_ori[0]
                pose.orientation.y = transformed_ori[1]
                pose.orientation.z = transformed_ori[2]
                pose.orientation.w = transformed_ori[3]
                """
                # Draw the cube
                if None not in result['projected_points']:
                    points2d = []
                    for pair in result['projected_points']:
                        points2d.append(tuple(pair))
                    print ('Drawing Cube')
                    draw.draw_cube(points2d, self.draw_colors[m])
                im.save("./torchdata/im_dope_detect.png")


def rotate_vector(vector, quaternion):
    q_conj = transformations.quaternion_conjugate(quaternion)
    vector = np.array(vector, dtype='float64')
    vector = np.append(vector, [0.0])
    vector = transformations.quaternion_multiply(q_conj, vector)
    vector = transformations.quaternion_multiply(vector, quaternion)
    return vector[:3]


def main():
    """Main routine to run DOPE"""
    dd = DopeMain()
    imPath = './resources/no_background_Color.png' 
    dd.image_callback(imPath) 
    
if __name__ == "__main__":
    main()
