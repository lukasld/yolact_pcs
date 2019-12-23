import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix

table_top_z_in_world = 0.736 + 0.057 / 2

# initial poses for box
X_WBox0 = RigidTransform.Identity()
X_WBox0.set_translation([0.55, -0.28, table_top_z_in_world])

# apply rotation
X_BBnew = RigidTransform()
X_BBnew.set_rotation(RollPitchYaw(0.3, 0.2 , 0.1).ToRotationMatrix())

# box ground truth pose
X_WBox = X_BBnew.multiply(X_WBox0)


def VisualizeTransformedScenePointCloud(X_MS: RigidTransform, scene_point_cloud, vis):
    # Apply the returned transformation to the observed samples to align the observed
    # point cloud with the ground truth point cloud.
    transformed_scene_point_cloud = X_MS.multiply(scene_point_cloud.T).T

    n_scene = len(scene_point_cloud)

    # Create a yellow meshcat point cloud for visualization.
    vis['transformed_scene'].set_object(
        g.PointCloud(transformed_scene_point_cloud.T,
                     make_meshcat_color_array(n_scene, 1.0, 1.0, 0.0),
                     size=0.003))


def SegmentBoxFromScene(scene_point_cloud):
    scene_point_cloud_cropped = np.zeros(scene_point_cloud.shape)
    point_count = 0
    for point in scene_point_cloud:
        x = point[0]
        y = point[1]
        z = point[2]
        if z >= table_top_z_in_world+0.005 and np.abs(x-0.55) <= 0.11 and np.abs(y+0.28) <= 0.08:
            scene_point_cloud_cropped[point_count] = point
            point_count += 1
    return scene_point_cloud_cropped[:point_count]


def GetBoxScenePointCloud():
    scene_point_cloud = np.load('./resources/scene_point_cloud.npy') 
    box_point_cloud_scene = SegmentBoxFromScene(scene_point_cloud)
    return X_BBnew.multiply(box_point_cloud_scene.T).T
    

def make_meshcat_color_array(N, r, g, b):
    '''
    Construct a color array to visualize a point cloud in meshcat

    Args:
        N: int. number of points to generate. Must be >= number of points in the
            point cloud to color
        r: float. The red value of the points, 0.0 <= r <= 1.0
        g: float. The green value of the points, 0.0 <= g <= 1.0
        b: float. The blue value of the points, 0.0 <= b <= 1.0

    Returns:
        3xN numpy array of the same color
    '''
    color = np.zeros((3, N))
    color[0, :] = r
    color[1, :] = g
    color[2, :] = b

    return color