

import csv

#from model_3d.generate_dataset import *
from model_3d import utils


def read_csv_data(f_name):
    with open(f_name) as f:
        reader = csv.reader(f)

        return utils.decode_camera_data(list(map(
                lambda dec_row: list(map(
                    lambda r: float(r),
                    dec_row[1:]
                )),
                reader
        ))[0])






cam_origin, frames_vectors, pitch_vectors = read_csv_data('./model_3d/dataset/data.csv')

pitch_corners = utils.get_pitch_corners(pitch_vectors)






from Geometry3D import *

get_point = lambda v: Point(v[0], v[1], v[2])


def get_screen_intersection(point):
    return get_intersection(
        Plane(
            get_point(frames_vectors[0]),
            get_point(frames_vectors[1]),
            get_point(frames_vectors[2]),
        ),
        point
    )

def get_pitch_intersection(point):
    return get_intersection(
        Plane(origin(),Vector(0,0,1)),
        point
    )


def get_intersection(plane, point):
    return tuple(intersection(
        Line(
            get_point(cam_origin),
            get_point(point)
        ),
        plane
    ))



frames_pitch_vectors = list(map(
    get_pitch_intersection,
    frames_vectors
))






from shapely.geometry import Polygon, mapping

def get_inner_section(y, x):
    get_vector2d_list = lambda lst: list(map(
        lambda v: (v[0], v[1]),
        lst
    ))

    x = Polygon(get_vector2d_list(x))
    y = Polygon(get_vector2d_list(y))

    intersection = x.intersection(y)
    if intersection.is_empty:
        return []
    else:
        return list(mapping(
            intersection
        )['coordinates'][0])


inner_section = get_inner_section(pitch_corners, frames_pitch_vectors)

inner_section_screen = list(map(
    get_screen_intersection,
    frames_vectors
))








import numpy as np
from math import cos, sin, radians


def trig(angle):
  r = radians(angle)
  return cos(r), sin(r)


def matrix_dot(l):
    if len(l) == 0:
        return np.identity(3)
    elif len(l) == 1:
        return l[0]
    else:
        return np.dot(
            l[0],
            matrix_dot(l[1:])
        )


def matrix(rotation, translation):

    xC, xS = trig(rotation[0])
    yC, yS = trig(rotation[1])
    zC, zS = trig(rotation[2])

    dX = translation[0]
    dY = translation[1]
    dZ = translation[2]

    Translate_matrix = np.array([[1, 0, 0, dX],
                               [0, 1, 0, dY],
                               [0, 0, 1, dZ],
                               [0, 0, 0, 1]])

    Rotate_X_matrix = np.array([[1, 0, 0, 0],
                              [0, xC, -xS, 0],
                              [0, xS, xC, 0],
                              [0, 0, 0, 1]])

    Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                              [0, 1, 0, 0],
                              [-yS, 0, yC, 0],
                              [0, 0, 0, 1]])

    Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                              [zS, zC, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

    return matrix_dot([
        Rotate_Z_matrix,
        Rotate_Y_matrix,
        Rotate_X_matrix,
        Translate_matrix
    ])


get_matrix = lambda vecs: np.matrix(list(map(
    lambda v: list(v) + [1],
    vecs
)))


def rm_last(l):
    l.pop()
    return l

get_vecs = lambda matrix: list(map(
    rm_last,
    matrix
))

#print(frames_vectors)
frames_vectors = get_matrix(frames_vectors)
#print(frames_vectors)


rotation = [5,8,9]
translation = [4, 4, 5]

m = matrix(rotation, translation)

print(frames_vectors)
#print(m)
frames_vectors = np.dot(frames_vectors, m)

print(frames_vectors)
#print(get_vecs(frames_vectors))

m = np.linalg.inv(m)
#print(m)
frames_vectors = np.dot(frames_vectors, m)
print(frames_vectors)
#print(get_vecs(frames_vectors))
