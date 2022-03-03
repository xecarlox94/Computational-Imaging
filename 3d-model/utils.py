from mathutils import *
from functools import reduce


enc_vec2d = lambda v: [v.x, v.y]
dec_vec2d = lambda l: Vector((l[0], l[1]))

enc_vec3d = lambda v: [v.x, v.y, v.z]
dec_vec3d = lambda l: Vector((l[0], l[1], l[2]))

enc_vec = lambda v: enc_vec2d(v) if (len(v) == 2) else enc_vec3d(v)
dec_vec = lambda l, lgth: dec_vec2d(l) if (lgth == 2) else dec_vec3d(l)

enc_marker = lambda m: enc_vec(m[0]) + enc_vec(m[1])
dec_marker = lambda l, length: (
        dec_vec(l[:length], length),
        dec_vec(l[length:], length)
    )

enc_bool = lambda x: 1 if x == True else 0
dec_bool = lambda x: True if x == 1 else False



enc_point = lambda enc_marker, dec_is_in: enc_marker + [enc_bool(dec_is_in)]

dec_point = lambda lst, dec_mrk: (dec_mrk, dec_bool(lst[len(lst) - 1]))


enc_3dpoint = lambda t: enc_point(
                enc_marker(t[0]),
                t[1]
            )

dec_3dpoint = lambda data, vec_len: dec_point(
                data,
                dec_marker(data[:len(data) - 1], vec_len)
            )



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


def decode_3dpoint_list(lst):
    return dec_list(
        lst,
        (lambda x: dec_3dpoint(x, 3)),
        7
    )


def encode_3dpoint_list(lst):
    return enc_list(
        (lambda t: enc_point(
            enc_marker(t[0]),
            t[1]
        )),
        lst
    )


def encode_marker_list(lst):
    return enc_list(
        enc_marker,
        lst
    )


def decode_marker_list(lst):
    return dec_list(
        lst,
        (lambda x: dec_marker(x, 3)),
        6
    )
