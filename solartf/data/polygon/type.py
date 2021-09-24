import numpy as np
import math, random


class Polygon:
    def __init__(self, x, y, avg_r, num_verts, angle_scale=0., irregularity=0., spikeyness=0.):
        """
        :param x, y: coordinates of the "centre" of the polygon
        :param avg_r: in px, the average radius of this polygon,
               this roughly controls how large the polygon is,
               really only useful for order of magnitude.
        :param angle_scale: [0,1] scale of initial angle
        :param irregularity: [0,1] indicating how much variance there is in the angular spacing of vertices.
                             [0,1] will map to [0, 2pi/numberOfVerts]
        :param spikeyness: [0,1] indicating how much variance there is in each vertex from
                           the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
        :param num_verts: self-explanatory
        """
        self.x = x
        self.y = y
        self.avg_r = avg_r
        self.num_verts = num_verts
        self.angle_scale = self._clip(angle_scale, 0, 1) * random.uniform(0, 2 * math.pi)
        self.irregularity = self._clip(irregularity, 0, 1) * 2 * np.pi / self.num_verts
        self.spikeyness = self._clip(spikeyness, 0, 1) * self.avg_r
        self.angle_steps = self._generate_angle_steps()
        self.points = self._generate_points()

    def _generate_angle_steps(self):
        angle_steps = []
        lower = (2 * np.pi / self.num_verts) - self.irregularity
        upper = (2 * np.pi / self.num_verts) + self.irregularity
        sum = 0
        for _ in range(self.num_verts):
            tmp = np.random.uniform(lower, upper)
            angle_steps.append(tmp)
            sum = sum + tmp

        # normalize the steps so that point 0 and point n+1 are the same
        k = sum / (2 * np.pi)
        return np.array(angle_steps) / k

    def _generate_points(self):
        points = np.zeros((self.num_verts, 2))
        angle = self.angle_scale
        for i in range(self.num_verts):
            r_i = self._clip(np.random.normal(self.avg_r, self.spikeyness), 0, 2 * self.avg_r)
            x = self.x + r_i * math.cos(angle)
            y = self.y + r_i * math.sin(angle)
            points[i] = np.array([x, y])

            angle = angle + self.angle_steps[i]

        return points.astype(np.int32)

    @staticmethod
    def _clip(x, min, max):
        if min > max:
            return x
        elif x < min:
            return min
        elif x > max:
            return max
        else:
            return x
