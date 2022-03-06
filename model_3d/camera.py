import math
from random import random


class Cam:

    def __init__(self, camera, index):
        self.direction = True
        self.change = 0.1
        self.camera = camera

        # to delete
        self.index = index
        self.reset_angle()



    def reset_angle(self):
        self.camera.rotation_euler[2] = math.radians(90)


    def get_angle(self):
        return self.camera.rotation_euler[2]

    def get_change(self):
        return self.change * random()

    def change_degrees(self, deg_change):
        new_angle = self.camera.rotation_euler[2] + deg_change
        self.camera.rotation_euler[2] = new_angle

    def change_angle(self):
        high_bound = math.radians(120)
        lower_bound = math.radians(60)

        curr_angle = self.get_angle()

        #print("id: " + str(self.index))
        #print(curr_angle)
        #print(math.degrees(curr_angle))

        if curr_angle >= high_bound:
            #print("changing to left")
            self.direction = False

        if curr_angle <= lower_bound:
            #print("changing to right")
            self.direction = True


        if self.direction == True:
            self.change_degrees(self.get_change())
            #print("going right")
        else:
            #print("going left")
            self.change_degrees(-self.get_change())

        #print("\n\n")
