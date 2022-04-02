
from functools import reduce


dec = lambda v: v * 100
enc = lambda v: v / 100


enc_vec2d = lambda v: [
    enc(v.x),
    enc(v.y)
]

dec_vec2d = lambda l: (
    dec(l[0]),
    dec(l[1])
)

enc_vec3d = lambda v: [
    enc(v.x),
    enc(v.y),
    enc(v.z)
]

dec_vec3d = lambda l: (
    dec(l[0]),
    dec(l[1]),
    dec(l[2])
)

enc_vec = lambda v: enc_vec2d(v) if (len(v) == 2) else enc_vec3d(v)
dec_vec = lambda l, lgth: dec_vec2d(l) if (lgth == 2) else dec_vec3d(l)


enc_list = lambda enc_fun, dec_list: reduce(
    lambda x, y: x + enc_fun(y),
    dec_list,
    []
)


def dec_list(data_list, decode_fun, unit_length):
    if len(data_list) == 0:
        return data_list

    data_to_decode = data_list[len(data_list) - unit_length:]
    decoded = decode_fun(data_to_decode)

    rest_of_data = data_list[:len(data_list) - unit_length]

    return dec_list(
            rest_of_data,
            decode_fun,
            unit_length
    ) + [decoded]


def encode_vector_list(lst):
    return enc_list(
        enc_vec,
        lst
    )


def decode_vector_list(lst, length):
    return dec_list(
        lst,
        lambda x: dec_vec(x, length),
        length
    )


def corner_vecs_pitch_vecs(pitch_vectors):
    corner_vectors = []
    p_corners = []

    for i in range(len(pitch_vectors)):
        if i in [0, 9, 29, 38]:
            corner_vectors.append(pitch_vectors[i])
        else:
            p_corners.append(pitch_vectors[i])

    return (
        corner_vectors,
        p_corners
    )


def get_pitch_corner_vecs(decoded_cam_data):
    pitch_vectors = decoded_cam_data[2]
    corner_vectors = []
    for i in [0, 9, 29, 38]:
        corner_vectors.append(pitch_vectors[i])
    return corner_vectors


def get_pitch_vectors(corner_vectors, pitch_vectors):
    i = 0
    for k in [0, 9, 29, 38]:
        pitch_vectors.insert(k, corner_vectors[i])
        i = i + 1
    return pitch_vectors


def encode_camera_data(camera_data_tuple):
    origin, frames_vectors, pitch_vectors = camera_data_tuple

    enc_origin = enc_vec(origin)
    enc_frames_vectors = encode_vector_list(frames_vectors)

    corner_vectors, pitch_vectors = corner_vecs_pitch_vecs(pitch_vectors)

    enc_corner_vectors = encode_vector_list(corner_vectors)
    enc_pitch_vectors = encode_vector_list(pitch_vectors)

    return (
        enc_origin +
        enc_frames_vectors +
        enc_corner_vectors +
        enc_pitch_vectors
    )


def decode_camera_data(enc_data):
    origin_enc_data = enc_data[:3]

    origin_decoded = dec_vec(origin_enc_data, 3)
    enc_data = enc_data[3: ]

    len_vectors_section = (3 * 4)
    frames_vectors_data = enc_data[ :len_vectors_section]
    frames_vectors_decoded = decode_vector_list(
            frames_vectors_data,
            3
    )
    enc_data = enc_data[len_vectors_section: ]

    len_vectors_section = (2 * 4)
    corner_vectors_data = enc_data[ :len_vectors_section]
    corner_vectors_decoded = decode_vector_list(
            corner_vectors_data,
            2
    )
    enc_data = enc_data[len_vectors_section: ]

    pitch_vectors_decoded = decode_vector_list(
            enc_data,
            2
    )

    pitch_vectors_decoded = get_pitch_vectors(
        corner_vectors_decoded,
        pitch_vectors_decoded
    )

    def get_pitch_vec(v, i):
        index = i + 1
        z = 0

        if index in [5, 7, 34, 36]: z = 2.4

        return (v[0], v[1], z)


    pitch_vectors_decoded = [
        get_pitch_vec(
            pitch_vectors_decoded[i],
            i
        ) for i in range(len(
            pitch_vectors_decoded
        ))
    ]

    return (
        origin_decoded,
        frames_vectors_decoded,
        pitch_vectors_decoded
    )



get_axis = lambda points, axis: [
    p[axis] for p in points
]


get_center = lambda points, axis: (
    sum(
        get_axis(points, axis)
    ) / len(points)
)


def get_2d_centroid(points):
    return (
        get_center(points, 0),
        get_center(points, 1),
    )


def get_3d_centroid(points):
    return (
        get_center(points, 0),
        get_center(points, 1),
        get_center(points, 2)
    )


