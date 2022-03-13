from mathutils import *
from functools import reduce


dec = lambda v: v * 100
enc = lambda v: v / 100


enc_vec2d = lambda v: [
    enc(v.x),
    enc(v.y)
]

dec_vec2d = lambda l: Vector((
    dec(l[0]),
    dec(l[1])
))

enc_vec3d = lambda v: [
    enc(v.x),
    enc(v.y),
    enc(v.z)
]

dec_vec3d = lambda l: Vector((
    dec(l[0]),
    dec(l[1]),
    dec(l[2])
))

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


def decode_vector_list(lst):
    return dec_list(
        lst,
        (lambda x: dec_vec(x, 2)),
        2
    )
