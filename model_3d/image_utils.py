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

    get_view_vector = lambda v: Vector((v.x, v.y, v.z))

    intersects = list(
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

    def get_camera_view(scn, o, vec):
        vec = world_to_camera_view(scn, o, vec)

        x = round(vec.x)
        y = round(vec.y)

        return Vector((x, y, 0.0))


    corner_frames = list(map(lambda v: get_camera_view(scn, o, v), frames))

    corner_vectors = list(zip(corner_frames, intersects))

    #cameraSquare = Square(corner_vectors)

    pitch_markers = list(
        map(
            lambda x: [x, [], False],
            range(1, 39 + 1)
        )
    )



    get_marker = lambda c, o: (o.name, (c, get_view_vector(o.location)))



    markers = []

    coll = bpy.data.collections.get("Markers")
    for obj in coll.all_objects:

        i = int(obj.name)


        vec = obj.location


        # Change to accomodate goal's corners (keep Z-index in it)
        # goal's top corners [5, 7, 34, 36] height is 2.4
        camview_vec = get_view_vector(world_to_camera_view(scn, o, vec))



        marker = get_marker(camview_vec, obj)
        markers.append(marker)


        vec_x = camview_vec[0]
        vec_y = camview_vec[1]
        # Get pitch points, within frame
        isInsideScreen = (vec_x >= 0 and vec_x <= 1) and (vec_y >= 0 and vec_y <= 1)



        pitch_markers[i-1][0] = int(marker[0])
        pitch_markers[i-1][1] = marker[1]
        pitch_markers[i-1][2] = isInsideScreen


    pitch_markers.sort(key=(lambda x: x[0]))

    pitch_markers = list(
            map(
                lambda x: (x[1], x[2]),
                pitch_markers
            )
    )


    return origin, corner_vectors, pitch_markers



def encode_camera_data(camera_data_tuple):
    origin, pitch_corners, pitch_markers = camera_data_tuple
    enc_origin = utils.enc_vec(origin)
    enc_corner_markers = utils.encode_marker_list(pitch_corners)
    enc_pitch_markers = utils.encode_3dpoint_list(pitch_markers)


    return (
            enc_origin +
            enc_corner_markers +
            enc_pitch_markers
    )


def decode_camera_data(enc_data):
    origin_enc_data = enc_data[:3]

    origin_decoded = utils.dec_vec(origin_enc_data, 3)
    enc_data = enc_data[3:]

    corner_markers_data = enc_data[: (6 * 4) ]
    corner_markers_decoded = utils.decode_marker_list(
            corner_markers_data
    )

    pitch_markers_data = enc_data[ (6 * 4) :]
    pitch_markers_decoded = utils.decode_3dpoint_list(
            pitch_markers_data
    )

    return (
        origin_decoded,
        corner_markers_decoded,
        pitch_markers_decoded
    )



def get_write_points(scn):

    coll = bpy.data.collections.get("Cameras")

    for o in coll.all_objects:

        enc_data = encode_camera_data(
                get_camera_data(o, scn)
        )

        decoded_data = decode_camera_data(enc_data)

        break

