import pdb

import numpy as np
from sklearn.neighbors import NearestNeighbors
import itertools
from helper_fns.point_cloud_processing import make_meshcat_color_array
from pydrake.math import RollPitchYaw
import meshcat

verbose = False

def test_on_points(vis, name, cloud, function, pts, radius=0.02):
    '''
    Args:
      name is a string
      cloud is (Nx3) numpy array
      function is a function of 2 args in pts are provided or of 1 arg if not
      pts is an array of indices into cloud
    Returns:
      None
    '''
    print(name, 'planes')
    print('Bounding box:', np.amin(cloud, axis=0), np.max(cloud, axis=0))

    # Selects neighbors within radius
    nbrs = NearestNeighbors(radius=radius).fit(cloud)
    # dists is a tuple of arrays of distances (to neighbors)  for each index in pts
    # indices is a tuple of arrays of indices (of neighbors) for each index in pts
    dists, indices = nbrs.radius_neighbors(cloud[pts])
    for i in range(pts.shape[0]):
        result = function(cloud, indices[i])
        if isinstance(result, tuple):
            # ransanc
            best_inliner = result[0]
            plane_equation = result[1]
            print("In/Outlier: ", best_inliner, "/", len(indices[i]), plane_equation)
        else:
            # plane fitting
            plane_equation=result
            print(plane_equation)
            
        DrawFacet(vis, plane_equation, center=np.mean(cloud[indices[i]], axis=0),
                  name=str(i), radius=radius, prefix=name + "_facets")


def fit_plane(xyzs):
    '''
    Args:
      xyzs is (N, 3) numpy array
    Returns:
      (4,) numpy array
    '''
    # your code here
    ##################
    center = np.mean(xyzs, axis=0)
    cxyzs = xyzs - center
    U, S, V = np.linalg.svd(cxyzs)
    normal = V[-1]              # last row of V
    d = -center.dot(normal)
    plane_equation = np.hstack([normal, d])
    ##################
    return plane_equation


def fit_plane_ransac(xyzs, tolerance, max_iterations):
    '''
    Args:
      xyzs is (N, 3) numpy array
      tolerance is a float
      max_iterations is a (small) integer
    Returns:
      (4,) numpy array
    '''
    best_ic = 0                 # inlier count
    best_model = None           # plane equation ((4,) array)
    N = xyzs.shape[0]           # number of points

    # your code here
    ##################
    sample_size = 3
    # "xyzs_1" is Nx4 (augmented with a 1 column)
    xyzs_1 = np.ones((N, 4))
    xyzs_1[:, :3] = xyzs
    for i in range(max_iterations):
        s = xyzs[np.random.randint(N, size=sample_size)]
        m = fit_plane(s)
        abs_distances = np.abs(np.dot(m, xyzs_1.T)) # 1 x N
        inliner_count = np.sum(abs_distances < tolerance)

        if verbose:
            print(i, '# inliers:', best_ic, 'best fit_plane:', best_model,)

        if inliner_count > best_ic:
            best_ic = inliner_count
            best_model = m
    ##################

    if verbose:
        print('best model:', best_model, 'explains:', best_ic)

    return  best_ic, best_model


def VoxelSubsample(cloud: np.array, voxel_size: float) -> np.array:
    """
    Args:
    cloud: (N, 3) numpy array representing a point cloud.
    voxel_size: the length of the edge of each voxel in meters.
    
    Returns:
    The sub-sampled point cloud as a (N_sub, 3) numpy array.
    """
    voxel_subsampler = dict()

    # your code here
    ####################################
    for xyz in cloud:
        x_int = int(xyz[0] / voxel_size)
        y_int = int(xyz[1] / voxel_size)
        z_int = int(xyz[2] / voxel_size)
        voxel_subsampler[(x_int, y_int, z_int)] = (xyz.tolist())
        
    return np.array(list(voxel_subsampler.values()))


def ComputeNormals(cloud: np.array, radius: float, num_points=None):
    """
    Args:
    cloud: (N, 3) numpy array representing a point cloud.
    radius: When estimating the normal at P, points inside a ball of
        this radius centered at P are used to estimate P's normal.
    num_points: if it is a integer, then the normal at (num_points) 
        randomly-chosen points in cloud are estimated using Ransac.
        If it is none, then the entire point cloud is chosen. 
    Returns:
    1. the normals at the chosen points in cloud as a (num_points, 3) 
        numpy array.
    2. the chosen points in cloud as a (num_points, 3) numpy array.
    """
    N = len(cloud)
    if num_points is None:
        num_points = N
        pts_idx = np.arange(N)
    else:
        pts_idx = np.random.randint(N, size=num_points)

    normals_at_pts = np.zeros((num_points, 3))
    center = np.mean(cloud, axis=0)

    # your code here
    ####################################
    # nearest neighbors of points selected for normal estimation.
    nbrs = NearestNeighbors(radius=radius).fit(cloud)
    dists, indices = nbrs.radius_neighbors(cloud[pts_idx])

    def flip_normal(center, xyz, normal):
        a = xyz - center
        a /= np.linalg.norm(a)
        if normal.dot(a) < 0:
            normal *= -1
        return normal.copy()

    for i in range(num_points):
        best_inliner, plane_equation = fit_plane_ransac(
            cloud[indices[i]], tolerance=1e-5, max_iterations=10)

        normal = plane_equation[:3]
        xyz = cloud[pts_idx[i]]
        normal_new = flip_normal(center, xyz, normal)
        normals_at_pts[i] = normal_new
    ####################################
    return normals_at_pts, cloud[pts_idx]


def DrawNormals(vis: meshcat.Visualizer,  name: str, normals, centers, radius):
    assert len(normals) == len(centers)
    prefix = name + "_facets"
    vis[prefix].delete()
    for i in range(len(normals)):
        plane_equation = np.hstack((normals[i], [0]))
        DrawFacet(vis, plane_equation, center=centers[i],
          name=str(i), radius=radius, prefix=prefix)


# visualize a facet
def DrawFacet(vis, abcd, name, center=None,
              prefix='facets', radius=0.02, thickness=0.001, color=0xffffff, opacity=0.6):
    normal = np.array(abcd[:3]).astype(float)
    normal /= np.linalg.norm(normal)
    d = -abcd[3] / np.linalg.norm(normal)

    R = np.eye(3)
    R[:, 2] = normal
    z = normal
    if abs(z[0]) < 1e-8:
        x = np.array([0, -normal[2], normal[1]])
    else:
        x = np.array([-normal[1], normal[0], 0])
    x /= np.linalg.norm(x)
    R[:, 0] = x
    R[:, 1] = np.cross(z, x)

    X = np.eye(4)
    Rz = RollPitchYaw(np.pi/2, 0, 0).ToRotationMatrix().matrix()
    X[:3, :3] = R.dot(Rz)
    if center is None:
        X[:3, 3] = d * normal
    else:
        X[:3, 3] = center
            
    
    X_normal = X.copy()
    X_normal[:3, :3] = R
    
    material = meshcat.geometry.MeshLambertMaterial(
        color=color, opacity=opacity)
    
    vis[prefix][name]["plane"].set_object(
        meshcat.geometry.Cylinder(thickness, radius), material)
    
    normal_vertices = np.array([[0, 0, 0], [0, 0, radius]]).astype(float)
    vis[prefix][name]["normal"].set_object(
        meshcat.geometry.Line(meshcat.geometry.PointsGeometry(normal_vertices.T)))
    
    vis[prefix][name]["plane"].set_transform(X)
    vis[prefix][name]["normal"].set_transform(X_normal)
    


    
    
