import numpy as np
from image_processing import line_point


def draw_to_image(image, points, color=[0, 255, 0], thick=False):
    for p in points:
        p = p.astype(int)
        if 0 <= p[1] < image.shape[0] and 0 <= p[0] < image.shape[1]:
            image[p[1], p[0]] = color
        if thick:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (0 <= p[1] + i < image.shape[0] and
                            0 <= p[0] + j < image.shape[1]):
                        image[p[1] + i, p[0] + j] = color


def line_points(cols, angles, dists):
    return np.array(
        [[line_point(0, angle, dist), line_point(cols, angle, dist)]
         for i, angle, dist in zip(range(len(angles)), angles, dists)])