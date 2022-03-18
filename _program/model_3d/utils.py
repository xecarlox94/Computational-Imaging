
from mathutils import Vector
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
        (lambda x: dec_vec(x, length)),
        length
    )


def encode_camera_data(camera_data_tuple):
    origin, center_screen, frames_vectors, pitch_vectors = camera_data_tuple

    enc_origin = enc_vec(origin)
    enc_center_screen = enc_vec(center_screen)
    enc_frames_vectors = encode_vector_list(frames_vectors)
    enc_pitch_vectors = encode_vector_list(pitch_vectors)

    return (
            enc_origin +
            enc_center_screen +
            enc_frames_vectors +
            enc_pitch_vectors
    )


def decode_camera_data(enc_data):
    origin_enc_data = enc_data[:3]

    origin_decoded = dec_vec(origin_enc_data, 3)
    enc_data = enc_data[3: ]

    center_screen_enc_data = enc_data[:3]

    center_screen_decoded = dec_vec(center_screen_enc_data, 3)
    enc_data = enc_data[3: ]

    len_corners_vectors = (2 * 4)
    frames_vectors_data = enc_data[: len_corners_vectors]
    frames_vectors_decoded = decode_vector_list(
            frames_vectors_data,
            2
    )

    enc_data = enc_data[len_corners_vectors :]
    pitch_vectors_decoded = decode_vector_list(
            enc_data,
            2
    )

    def get_pitch_vec(v, i):
        z = 0

        index = i + 1
        if index in [5, 7, 34, 36]:
            z = 2.4

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
        center_screen_decoded,
        frames_vectors_decoded,
        pitch_vectors_decoded
    )




def get_pitch_corners(pitch_vectors):
    pitch_corners = dict()

    for i in range(len(pitch_vectors)):
        ix = i + 1

        if ix in [1, 10, 30, 39]:
            pitch_corners[str(ix)] = pitch_vectors[i]

    return [
        pitch_corners["39"],
        pitch_corners["30"],
        pitch_corners["1"],
        pitch_corners["10"]
    ]

