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

    get_view_vector = lambda v: Vector((v.x, v.y))

    origin = matrix.to_translation()

    pitch_vectors = list(map(
        lambda m_obj: (
            int(m_obj.name),
            get_view_vector(m_obj.location)
        ),
        bpy.data.collections.get("Markers").all_objects
    ))

    pitch_vectors.sort(key=(lambda x: x[0]))

    pitch_vectors = list(map(
        lambda x: x[1],
        pitch_vectors
    ))

    frames_vectors = [matrix @ v for v in camera.view_frame(scene=scn)]

    frames_pitch_vectors = list(map(
        lambda v: get_view_vector(
            intersect_line_plane(
                origin,
                v,
                Vector((0, 0, 0)),
                Vector((0, 0, 1)),
            )
        ),
        frames_vectors
    ))

    """
    get_v = lambda v: (int(v.x), int(v.y), int(v.z))
    print(list(map(
        lambda v: get_v(
            world_to_camera_view(scn, o, v)
        ),
        frames_vectors
    )))
    """

    p1 = frames_vectors[0]
    p2 = frames_vectors[1]
    p3 = frames_vectors[2]
    plane_no = geometry.normal([p1, p2, p3])

    """
    d = geometry.distance_point_to_plane(origin, origin, plane_no)
    print(round(d, 4))
    """



    get_axis = lambda points, axis: [p[axis] for p in points]
    get_center = lambda points, axis: (
        sum(
            get_axis(points, axis)
        ) / len(points)
    )

    def get_2d_centroid(points):
        return Vector((
            get_center(points, 0),
            get_center(points, 1),
        ))

    def get_3d_centroid(points):
        return Vector((
            get_center(points, 0),
            get_center(points, 1),
            get_center(points, 2)
        ))


    pitch_corners = utils.get_pitch_corners(pitch_vectors)

    center_screen = get_3d_centroid(frames_vectors)


    return (
        origin,
        center_screen,
        frames_pitch_vectors,
        pitch_vectors
    )
