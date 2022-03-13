import bpy

from bpy_extras.object_utils import world_to_camera_view

from mathutils.geometry import intersect_line_plane

from mathutils import *


import sys
import os

cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd)


import utils



def render_image(scene, cam, file_name):
    bpy.context.scene.camera = cam

    scene.render.image_settings.file_format = 'JPEG'

    scene.render.filepath = "./dataset/" + file_name

    scene.render.resolution_x = 256
    scene.render.resolution_y = 256

    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    bpy.ops.render.render(write_still = True)



def get_camera_data(o, scn):
    camera = o.data
    matrix = o.matrix_world.normalized()

    frames = [matrix @ v for v in camera.view_frame(scene=scn)]

    origin = matrix.to_translation()

    get_view_vector = lambda v: Vector((v.x, v.y))


    corner_vectors  = list(
        map(
            lambda f: get_view_vector(
                intersect_line_plane(
                    origin,
                    f,
                    Vector((0, 0, 0)),
                    Vector((0, 0, 1))
                )
            ),
            frames
        )
    )

    pitch_vectors = list(map(
        lambda m_obj: (
            int(m_obj.name),
            get_view_vector(
                world_to_camera_view(
                    scn,
                    o,
                    m_obj.location
                )
            )
        ),
        bpy.data.collections.get("Markers").all_objects
    ))


    # Change to accomodate goal's corners (keep Z-index in it)
    # goal's top corners [5, 7, 34, 36] height is 2.4
    pitch_vectors.sort(key=(lambda x: x[0]))


    pitch_vectors = list(
            map(
                lambda x: x[1],
                pitch_vectors
            )
    )

    return origin, corner_vectors, pitch_vectors



def encode_camera_data(camera_data_tuple):
    origin, corner_vectors, pitch_vectors = camera_data_tuple
    enc_origin = utils.enc_vec(origin)
    enc_corner_vectors = utils.encode_vector_list(corner_vectors)
    enc_pitch_vectors = utils.encode_vector_list(pitch_vectors)


    return (
            enc_origin +
            enc_corner_vectors +
            enc_pitch_vectors
    )


def decode_camera_data(enc_data):
    origin_enc_data = enc_data[:3]

    origin_decoded = utils.dec_vec(origin_enc_data, 3)
    enc_data = enc_data[3:]


    len_corner_vectors = (2 * 4)

    corner_vectors_data = enc_data[: len_corner_vectors ]
    corner_vectors_decoded = utils.decode_vector_list(
            corner_vectors_data
    )

    pitch_vectors_data = enc_data[ len_corner_vectors :]
    pitch_vectors_decoded = utils.decode_vector_list(
            pitch_vectors_data
    )

    return (
        origin_decoded,
        corner_vectors_decoded,
        pitch_vectors_decoded
    )
