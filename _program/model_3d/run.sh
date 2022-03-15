#!/bin/sh

#blender --debug-all -b pitch.blend -P script.py

#blender -b pitch.blend -P script.py

#blender -b pitch.blend -P test.py

#blender -b pitch.blend -P utils.py

blender -b pitch.blend -P generate_dataset.py
