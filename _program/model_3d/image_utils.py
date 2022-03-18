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


    frames_vectors = [matrix @ v for v in camera.view_frame(scene=scn)]


    origin = matrix.to_translation()


    get_view_vector = lambda v: Vector((v.x, v.y))


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

    return (
        origin,
        frames_vectors,
        pitch_vectors
    )


