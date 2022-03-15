import bpy
import os
import math

def get_filename(name):
    filepath = bpy.data.filepath
    directory = os.path.dirname(filepath)
    return os.path.join( directory , name)

def run_file(filename):
    filename = get_filename(filename)
    exec(compile(open(filename).read(), filename, 'exec'))
