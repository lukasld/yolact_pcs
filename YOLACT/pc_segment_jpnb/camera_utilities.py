import numpy as np
import yaml

from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw

with open("./resources/camera_info.yaml", "r") as file:
    camera_info = yaml.safe_load(file)


#Camera Parameters in pixels
camera_width = camera_info["camera_intrinsics"]["width"]
camera_height = camera_info["camera_intrinsics"]["height"]
focal_x = camera_info["camera_intrinsics"]["focal_x"]
focal_y = camera_info["camera_intrinsics"]["focal_y"]
center_x = camera_info["camera_intrinsics"]["center_x"]
center_y = camera_info["camera_intrinsics"]["center_y"]
    
#Transform from camera to world (c2w)
c2w_x = camera_info["world_transform"]["x"]
c2w_y = camera_info["world_transform"]["y"]
c2w_z = camera_info["world_transform"]["z"]
c2w_roll = camera_info["world_transform"]["roll"]
c2w_pitch = camera_info["world_transform"]["pitch"]
c2w_yaw = camera_info["world_transform"]["yaw"]


#usually S is 0
skew = 0.0

def calc_intrinsics():
    """ output the 3x3 intrinsics matrix as a (3,3) numpy array."""
    K = np.zeros((3, 3))
    ###### Fill in the Code ###########

    # we make the intrinsics camera building the 3,3 matrix
    K = np.array([[focal_x, skew, center_x],
                        [0, focal_y, center_y],
                        [0,   0, 1]]) 
    ###################################
    return K
    

def calc_extrinsics():
    """Returns the camera pose in the world frame as a RigidTransform"""

    ############ Fill in the code ##################

    rpy = RollPitchYaw([c2w_roll,c2w_pitch,c2w_yaw])
    T= np.array([c2w_x,c2w_y,c2w_z])
    X_WC = RigidTransform(rpy,T)

    ################################################
    return X_WC
