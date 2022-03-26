import sys
import os
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cwd)


import bpy
from camera import Cam
import utils
import image_utils

import csv


get_collection = lambda name: bpy.data.collections.get(name)
get_cam = lambda cam_collection, index: cam_collection.all_objects[index]



scene = bpy.context.scene
collection = get_collection("Cameras")
cam = get_cam(collection, 0)



cams = []
all_cams = collection.all_objects
for i in range(len(all_cams)):
    cams.append(Cam(all_cams[i], i))


enc_row = lambda img_str, enc_data: [img_str] + enc_data
dec_row = lambda enc_row: (
                enc_row[:1][0],
                list(
                    map(
                        lambda r: float(r),
                        enc_row[1:]
                    )
                )
            )


file_name = "./dataset/data.csv"
f = open(file_name, "x")


writer = csv.writer(f)

for loop_id in range(250):
    for cam_id in range(len(cams)):
        img_str = str(cam_id) + "_" + str(loop_id) + ".jpeg"


        image_utils.render_image(
            scene,
            cams[cam_id].camera,
            img_str
        )


        enc_data = utils.encode_camera_data(
            image_utils.get_camera_data(
                cams[cam_id].camera,
                scene
            )
        )


        writer.writerow(
            enc_row(
                img_str,
                enc_data
            )
        )


        cams[cam_id].change_angle()


f.close()



def read_csv_data(f_name):
    with open(f_name) as f:
        reader = csv.reader(f)

        for row in reader:
            print(dec_row(row))

#read_csv_data(file_name)




for i in range(len(all_cams)):
    cams[i].reset_angle()
