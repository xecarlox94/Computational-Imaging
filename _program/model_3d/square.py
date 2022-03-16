
class Square:

    def __init__(self, points):
        self.vector_tuple = lambda v: (v.x, v.y)
        self.index_point = lambda i: self.points[1][i]
        self.gpitch_vector = lambda p: p[1]

        self.points = []

        for i in range(len(points)):
            nxt = ((i + 1) % len(points))

            prv = i - 1
            if prv < 0: prv = len(points)

            self.points.append(
                (
                    points[i],
                    (prv, nxt)
                )
            )



    def is_point_inside(self, point):

        def is_on_right_side(x, y, xy0, xy1):
            x0, y0 = xy0
            x1, y1 = xy1
            a = float(y1 - y0)
            b = float(x0 - x1)
            c = - a*x0 - b*y0
            return a*x + b*y + c >= 0

        isInside = True
        for i in range(len(self.points)):
            point = self.gpitch_vector(
                self.points[i]
            )

            curr_point = self.vector_tuple(point[0])


            next_point = self.vector_tuple(
                self.index_point(point[1][1])[0]
            )

            is_right = is_on_right_side(
                point.x,
                point.y,
                self.gpitch_vector(curr_point),
                self.gpitch_vector(next_point)
            )

            isInside = isInside & is_right

        return isInside


    def get_inner_section(self, square):

        is_inside_points = list(
            map(
                self.is_point_inside,
                square.points
            )
        )

        are_all_true = reduce(
            lambda x,y: x & y,
            is_inside_points
        )

        are_all_false = True
        for is_in in is_inside_points:
            if is_in == True:
                are_all_false = False
                break


        if are_all_true == True:
            return square.points

        elif are_all_false == True:
            return self.points

        else:

            for i in range(len(self.points)):

                point = self.points[i]
                next_point = self.points[i + 1]

                opoint = square.points[i]
                next_opoint = square.points[i + 1]

                is_inside = is_inside_points[i]
                is_next_inside = is_inside_points[i + 1]


                """
                if is_inside == True:
                    if is_next_inside == True:
                        #ajksdhaskd

                    else:
                        #ajksdhaskd


                else:
                    if is_next_inside == True:
                        #ajksdhaskd

                    else:
                        #ajksdhaskd

                """
