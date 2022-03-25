import csv
import numpy as np

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






from Geometry3D import Plane, Point, Line, origin, Vector, intersection, y_unit_vector

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



from scipy.spatial.transform import Rotation as R

def get_rotation(v2, v1):
    gv = lambda v: np.reshape(
        np.array(v),
        (1, -1)
    )

    return tuple(R.align_vectors(
        gv(v2),
        gv(v1)
    )[0].as_euler('xyz'))


def get_frame_vec_rotation(f_centroid_points):
    origin = get_point(f_centroid_points[0])
    top = get_point(f_centroid_points[1])

    return get_rotation(
        tuple(
            Vector(
                origin,
                top
            )
        ),
        tuple(y_unit_vector())
    )



from math import cos, sin



def matrix_dot(l):
    if len(l) == 1:
        return l[0]
    else:
        return np.dot(
            l[0],
            matrix_dot(l[1:])
        )





def matrix(translation, rotation):
    trig = lambda radians: (
        cos(radians),
        sin(radians)
    )


    xC, xS = trig(rotation[0])
    yC, yS = trig(rotation[1])
    zC, zS = trig(rotation[2])


    dX = translation[0]
    dY = translation[1]
    dZ = translation[2]


    translate_matrix = np.array([
        [ 1  , 0  , 0  , dX ],
        [ 0  , 1  , 0  , dY ],
        [ 0  , 0  , 1  , dZ ],
        [ 0  , 0  , 0  , 1  ]
    ])

    rotate_X_matrix = np.array([
        [ 1  , 0  , 0  , 0  ],
        [ 0  , xC , -xS, 0  ],
        [ 0  , xS , xC , 0  ],
        [ 0  , 0  , 0  , 1  ]
    ])

    rotate_Y_matrix = np.array([
        [ yC , 0  , yS , 0  ],
        [ 0  , 1  , 0  , 0  ],
        [ -yS, 0  , yC , 0  ],
        [ 0  , 0  , 0  , 1  ]
    ])

    rotate_Z_matrix = np.array([
        [ zC , -zS, 0  , 0  ],
        [ zS , zC , 0  , 0  ],
        [ 0  , 0  , 1  , 0  ],
        [ 0  , 0  , 0  , 1  ]
    ])

    return translate_matrix

    """
    return matrix_dot([
        translate_matrix,
        rotate_Z_matrix,
        rotate_Y_matrix,
        rotate_X_matrix
    ])
    """



get_matrix = lambda vecs: np.matrix(list(map(
    lambda v: list(v) + [1],
    vecs
)))


get_vectors = lambda matrix: list(map(
    lambda r: tuple(
        np.delete(r, -1)
    ),
    matrix.getA()
))


get_top_frames = lambda frames_vecs: (frames_vecs[0], frames_vecs[3])


get_frame_origin_vec = lambda frames_vecs: [
    utils.get_3d_centroid(frames_vecs),
    utils.get_3d_centroid(get_top_frames(frames_vecs))
]



frame_origin_vec = get_frame_origin_vec(frames_vectors)


centroid_vec_rotation = get_frame_vec_rotation(
    frame_origin_vec
)


get_translation = lambda vector: tuple(
    -np.array(vector)
)


print(frame_origin_vec[0])
print(get_translation(frame_origin_vec[0]))


m = matrix(
    get_translation(frame_origin_vec[0]),
    get_frame_vec_rotation(frame_origin_vec),
)


frame_origin_vec = get_matrix(frame_origin_vec)


frame_origin_vec = matrix_dot([
    frame_origin_vec,
    m
])


print(get_vectors(frame_origin_vec))


m_inverted = np.linalg.inv(m)


frame_origin_vec = matrix_dot([
    frame_origin_vec,
    m_inverted
])

