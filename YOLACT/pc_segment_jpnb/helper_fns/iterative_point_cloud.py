# Pset 2a: Iterative Closest Point
#Code adapted from Clay Flannigan
import numpy as np

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from sklearn.neighbors import NearestNeighbors
from pydrake.math import RigidTransform, RotationMatrix

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


def visualize_icp(meshcat_vis, scene, model, X_MS, marker_size=0.01):
    '''
    Visualize the ground truth (red), scene (blue), and transformed
    (yellow) point clouds in meshcat.

    Args:
        meschat_vis: an instance of a meshcat visualizer
        scene: an Nx3 numpy array representing scene point cloud
        model: an Mx3 numpy array representing model point cloud
        X_MS: RigidTransform from the scene point cloud to the model point cloud.
    '''

    meshcat_vis['model'].delete()
    meshcat_vis['observations'].delete()
    meshcat_vis['transformed_observations'].delete()

    # Make meshcat color arrays.
    N = scene.shape[0]
    M = model.shape[0]

    red = make_meshcat_color_array(M, 0.5, 0, 0)
    blue = make_meshcat_color_array(N, 0, 0, 0.5)
    yellow = make_meshcat_color_array(N, 1, 1, 0)

    # Create red and blue meshcat point clouds for visualization.
    model_meshcat = g.PointCloud(model.T, red, size=marker_size)
    observations_meshcat = g.PointCloud(scene.T, blue, size=marker_size)

    meshcat_vis['model'].set_object(model_meshcat)
    meshcat_vis['scene'].set_object(observations_meshcat)

    # Apply the returned transformation to the scene samples to align the
    # scene point cloud with the ground truth point cloud.
    transformed_scene = X_MS.multiply(scene.T)

    # Create a yellow meshcat point cloud for visualization.
    transformed_scene_meshcat = \
        g.PointCloud(transformed_scene, yellow, size=marker_size)

    meshcat_vis['transformed_scene'].set_object(
        transformed_scene_meshcat)


def clear_vis(meshcat_vis):
    '''
    Removes model, observations, and transformed_observations objects
    from meshcat.

    Args:
        meschat_vis: an instance of a meshcat visualizer
    '''

    meshcat_vis['model'].delete()
    meshcat_vis['observations'].delete()
    meshcat_vis['transformed_observations'].delete()


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

    distances = np.zeros(point_cloud_A.shape[0])
    indices = np.zeros(point_cloud_A.shape[0])

    # your code here
    ##################
    for i, p in enumerate(point_cloud_A):
        #d = np.linalg.norm(point_cloud_B - p, ord=2, axis=1)
        d = np.sqrt(np.sum((point_cloud_B - p) ** 2, 1))
        d_min, d_min_idx = np.min(d), np.argmin(d)
        distances[i] = d_min
        indices[i] = d_min_idx

    ##################

    return distances, np.array(indices, dtype=int)


def least_squares_transform(point_cloud_A, point_cloud_B) -> RigidTransform:
    '''
    Calculates the least-squares best-fit transform that maps corresponding
    points point_cloud_A to point_cloud_B.

    Args:
      point_cloud_A: Nx3 numpy array of corresponding points
      point_cloud_B: Nx3 numpy array of corresponding points
    Returns:
      X_BA: A RigidTransform object that maps point_cloud_A on to point_cloud_B
            such that
                        X_BA.multiply(point_cloud_A) ~= point_cloud_B,
    '''

    # number of dimensions
    m = 3
    X_BA = RigidTransform()

    # your code here
    ##################
    # A is scene, B is model

    mu_s, mu_m = np.mean(point_cloud_A, 0), np.mean(point_cloud_B, 0)
    w = 0
    for p_a, p_b in zip(point_cloud_A, point_cloud_B):
        w += np.outer(p_b - mu_m, p_a - mu_s)

    u, s, vh = np.linalg.svd(w)

    r = np.matmul(u, vh)
    if np.linalg.det(r) == -1:
        #print('determinant of r is -1')
        vh[:, 2] = -vh[:, 2]
        r = np.matmul(u, vh)

    t = mu_m - (np.dot(r, mu_s))
    rt = np.zeros((4, 4))
    rt[:3, :3] = r
    rt[:3, 3] = t
    rt[3, 3] = 1.0
    X_BA = RigidTransform(pose=rt)
    ##################

    return X_BA

def icp_w_perfect_correspondance(point_cloud_A, point_cloud_B,
        init_guess=None, max_iterations=20, tolerance=1e-3):
    '''
    The Iterative Closest Point algorithm: finds best-fit transform that maps
        point_cloud_A on to point_cloud_B. point_cloud_A[i] needs to correspond to
        point_cloud_B[i].

    Args:
        point_cloud_A: Nx3 numpy array of points to match to point_cloud_B
        point_cloud_B: Nx3 numpy array of points
        init_guess: homogeneous transformation representing an initial guess
            of the transform. If one isn't provided, the Idnetity transformation
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

    point_cloud_A0 = point_cloud_A.copy()

    # Number of dimensions
    m = 3

    # your code here
    ##################
    prev_error = 9999
    curr_error = 9998
    num_iters = 0

    # Using faster NN algorithm
    tracked_transform = X_BA.matrix().copy()

    while (prev_error - curr_error) > tolerance and num_iters < max_iterations:
        # Find the nearest neighbors
        err = np.sqrt(np.sum((point_cloud_A0 - point_cloud_B) ** 2, 1))
        nn_points = point_cloud_B

        # Compute the transformation from the current nns
        X_BA = least_squares_transform(point_cloud_A0, nn_points)
        x_ba = X_BA.matrix()

        point_cloud_A0 = np.concatenate([point_cloud_A0, np.ones((point_cloud_A.shape[0], 1))], 1)
        point_cloud_A0 = np.matmul(x_ba, point_cloud_A0.T).T
        point_cloud_A0 = point_cloud_A0[:, :3] / point_cloud_A0[:, 3:]
        tracked_transform = np.dot(x_ba, tracked_transform)#np.dot(tracked_transform, x_ba)

        prev_error = curr_error
        curr_error = np.mean(err)
        print(num_iters, curr_error)
        num_iters += 1

    X_BA = RigidTransform(pose=tracked_transform)
    mean_error = curr_error

    ###################

    return X_BA, mean_error, num_iters


def icp(point_cloud_A, point_cloud_B,
        init_guess=None, max_iterations=20, tolerance=1e-3):
    '''
    The Iterative Closest Point algorithm: finds best-fit transform that maps
        point_cloud_A on to point_cloud_B

    Args:
        point_cloud_A: Nx3 numpy array of points to match to point_cloud_B
        point_cloud_B: Nx3 numpy array of points
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

    point_cloud_A0 = point_cloud_A.copy()

    # Number of dimensions
    m = 3

    # your code here
    ##################
    if init_guess is not None:
        X_BA = init_guess #RigidTransform(pose=init_guess)
        point_cloud_A0 = X_BA.multiply(point_cloud_A0.T).T
        #X_BA = RigidTransform(pose=np.eye(4))
    else:
        X_BA = RigidTransform(np.eye(4))

    prev_error = 9999
    curr_error = 9998
    num_iters = 0
    from sklearn.neighbors import NearestNeighbors

    # Using faster NN algorithm
    nnbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(point_cloud_B) # , metric='l1'
    tracked_transform = X_BA#.matrix().copy()

    while (prev_error - curr_error) > tolerance and num_iters < max_iterations:
        # Find the nearest neighbors
        err, c = nnbrs.kneighbors(point_cloud_A0) #err, c = nearest_neighbors(point_cloud_A0, point_cloud_B)
        nn_points = np.array(point_cloud_B)[np.array(c)].squeeze(1)

        # Compute the transformation from the current nns
        X_BA = least_squares_transform(point_cloud_A0, nn_points)
        point_cloud_A0 = X_BA.multiply(point_cloud_A0.T).T

        # Update the transform
        tracked_transform = X_BA.multiply(tracked_transform)

        prev_error = curr_error
        curr_error = np.mean(err)
        #print(num_iters, curr_error)
        num_iters += 1

    X_BA = RigidTransform(pose=tracked_transform)
    mean_error = curr_error
    ###################

    return X_BA, mean_error, num_iters


def repeat_icp_until_good_fit(point_cloud_A,
                              point_cloud_B,
                              error_threshold,
                              max_tries,
                              init_guess=None,
                              max_iterations=20,
                              tolerance=0.001):
    '''
    Run ICP until it converges to a "good" fit.

    Args:
        point_cloud_A: Nx3 numpy array of points to match to point_cloud_B
        point_cloud_B: Nx3 numpy array of points
        error_threshold: float. maximum allowed mean ICP error before stopping
        max_tries: int. stop running ICP after max_tries if it hasn't produced
            a transform with an error < error_threshold.
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
        num_runs: int. total number of times ICP ran - not the total number of
            ICP iterations.
    '''

    # Transform from point_cloud_B to point_cloud_A
    # Overwrite this with ICP results.
    X_BA = RigidTransform()

    mean_error = np.inf
    num_runs = 0

    # your code here
    ##################

    Ts, all_errors, all_iters = [], [], []
    for num_runs in range(max_tries):
        print(num_runs)
        # apply random transformation to the target point_cloud_B
        if num_runs > 0:
            r = RigidTransform(meshcat.transformations.random_rotation_matrix())
            point_cloud_B0 = r.multiply(point_cloud_B.T).T
        else:
            point_cloud_B0 = point_cloud_B


        X_MS, mean_error, num_iters = icp(point_cloud_A, point_cloud_B0, init_guess=init_guess, max_iterations=max_iterations, tolerance=tolerance)

        # log results
        if num_runs > 0:
            rt = RigidTransform(np.linalg.inv(r.matrix()))
            Ts.append(rt.multiply(X_MS)) #.matrix()))
            #Ts.append(X_MS.multiply(rt))
        else:
            Ts.append(X_MS)

        all_errors.append(mean_error)
        all_iters.append(num_iters)

        if mean_error < error_threshold:
            break

    best_run = np.argmin(all_errors)
    #print(best_run)
    X_BA = RigidTransform(pose=Ts[best_run])
    #print(all_errors)
    mean_error = all_errors[best_run]
    ##################

    return X_BA, mean_error, num_runs