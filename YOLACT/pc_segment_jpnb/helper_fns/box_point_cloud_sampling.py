import numpy as np

def get_point_from_bottom(l, w, h):
    x = np.random.rand() * l - l/2
    y = np.random.rand() * w - w/2
    z = 0
    return np.array([x, y, z])

def get_point_from_top(l, w, h):
    x = np.random.rand() * l - l/2
    y = np.random.rand() * w - w/2
    z = h
    return np.array([x, y, z])

def get_point_from_front(l, w, h):
    x = l / 2
    y = np.random.rand() * w - w/2
    z = np.random.rand() * h
    return np.array([x, y, z])

def get_point_from_back(l, w, h):
    x = - l / 2
    y = np.random.rand() * w - w/2
    z = np.random.rand() * h
    return np.array([x, y, z])

def get_point_from_left(l, w, h):
    x = np.random.rand() * l - l/2
    y = -w/2
    z = np.random.rand() * h
    return np.array([x, y, z])

def get_point_from_right(l, w, h):
    x = np.random.rand() * l - l/2
    y = w/2
    z = np.random.rand() * h 
    return np.array([x, y, z])


def GetBoxModelPointCloud(num_points, l=0.2, w=0.15, h=0.15):
    '''
    This function uniformly samples points from the faces of a box.
    The number of points on each surface is proportional to the area of that surface.
    num_points: integer
        Number of points in the returned point cloud.
    l, w, h: length, width and height of the box in meters.
    return value: 
        np array of shape (3, num_points) 
    '''

    get_points = [get_point_from_bottom,
                  get_point_from_top,
                  get_point_from_front,
                  get_point_from_back,
                  get_point_from_left,
                  get_point_from_right]

    num_faces = len(get_points)

    areas = np.array([l * w,
                      l * w,
                      w * h,
                      w * h,
                      l * h,
                      l * h])

    normalized_areas = areas / areas.sum()
    weights = np.zeros(num_faces + 1)
    for i in range(num_faces):
        weights[i + 1] = weights[i] + normalized_areas[i]

    points = np.zeros((num_points, 3))

    for i in range(num_points):
        seed = np.random.rand()
        idx_face = 0
        while True:
            if seed > weights[idx_face] and seed < weights[idx_face + 1]:
                break
            idx_face += 1
        points[i] = get_points[idx_face](l, w, h)

    return points
