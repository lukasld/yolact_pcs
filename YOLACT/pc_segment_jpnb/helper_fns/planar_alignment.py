# Pset 2a: Iterative Closest Point
#Code adapted from Clay Flannigan
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pydrake.math import RigidTransform, RotationMatrix


def nearest_neighbors(point_cloud_A: np.array, point_cloud_B: np.array):
    '''
    Find the nearest (Euclidean) neighbor in point_cloud_B for each
    point in point_cloud_A.

    Args:
        point_cloud_A: Nx3 numpy array of points
        point_cloud_B: Mx3 numpy array of points
    Returns:
        distances: (N, ) numpy array of Euclidean distances from each point in
            point_cloud_A to its nearest neighbor in point_cloud_B.
        indices: (N, ) numpy array of the indices in point_cloud_B of each
            point_cloud_A point's nearest neighbor - these are the c_i's
    '''

    distances = np.zeros(point_cloud_A.shape[1])
    indices = np.zeros(point_cloud_A.shape[1])

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(point_cloud_B)
    distances, indices = nbrs.kneighbors(point_cloud_A)
    distances = distances.squeeze()
    indices = indices.squeeze()

    return distances, indices


def least_squares_transform(normals_A, normals_B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding
    points normals_A to normals_B.

    Args:
      normals_A: Nx3 numpy array of normals of point cloud A.
      normals_B: Nx3 numpy array of normals of point cloud B. 
    Returns:
      R_BA: A RotationMatrix object that maps normals_A on to normals_B 
            such that
                R_BA.multiply(normals_A) ~= normals_B,
    '''

    # number of dimensions
    m = 3
    R_BA = RotationMatrix()

    # your code here
    ###############################
    scene = normals_A
    model = normals_B

    W = model.T.dot(scene)
    U, Sigma, Vh = np.linalg.svd(W)
    R_star = U.dot(Vh)

    if np.linalg.det(R_star) < 0:
        Vh[-1] *= -1
        R_star = U.dot(Vh)

    R_BA = RotationMatrix(R_star)
    ###############################

    return R_BA


def align_planes_with_normals(point_cloud_A, point_cloud_B, normals_A, normals_B,
        init_guess=None, max_iterations=20, tolerance=1e-3):
    '''
    The Iterative Closest Point algorithm: finds best-fit transform that maps
        point_cloud_A on to point_cloud_B

    Args:
        point_cloud_A: Nx3 numpy array of points to match to point_cloud_B
        point_cloud_B: Nx3 numpy array of points
        normals_A: Nx3 numpy array of normals of point cloud A.
        normals_B: Nx3 numpy array of normals of point cloud B. 
        init_guess: 4x4 homogeneous transformation representing an initial guess
            of the transform. If one isn't provided, the 4x4 identity matrix
            will be used.
        max_iterations: int. if the algorithm hasn't converged after
            max_iterations, exit the algorithm
        tolerance: float. the maximum difference in the error between two
            consecutive iterations before stopping
    
    Returns:
        X_BA: A RigidTransform object that maps point_cloud_A on to point_cloud_B
              such that
                        X_BA.multiply(point_cloud_A) ~= point_cloud_B,

        mean_error: float. mean of the Euclidean distances from each point in
            the transformed point_cloud_A to its nearest neighbor in
            point_cloud_B
        num_iters: int. total number of iterations run
    '''

    # Transform from point_cloud_B to point_cloud_A
    # Overwrite this with ICP results.
    X_BA = RigidTransform()
    mean_error = 0
    num_iters = 0

    # Apply initial guess
    if init_guess is not None:
        X_BA = RigidTransform(init_guess)
        point_cloud_A = X_BA.multiply(point_cloud_A.T).T

    # Your code here
    ###################################################
    # Look for rotations in a loop.
    prev_error = 0
    R_BA = X_BA.rotation()
    while True:
        num_iters += 1

        distances, indices = nearest_neighbors(normals_A, normals_B)

        R_BA_step = least_squares_transform(normals_A, normals_B[indices])

        R_BA = R_BA_step.multiply(R_BA)

        normals_A = R_BA_step.multiply(normals_A.T).T

        mean_error = np.mean(distances)

        if abs(mean_error - prev_error) < tolerance or num_iters >= max_iterations:
            break

        prev_error = mean_error

    X_BA.set_rotation(R_BA)

    # Compute once for translation
    mu_A = np.mean(point_cloud_A, axis=0)
    mu_B = np.mean(point_cloud_B, axis=0)
    X_BA.set_translation(mu_B - R_BA.multiply(mu_A))
    ###################################################
    return X_BA, mean_error, num_iters

