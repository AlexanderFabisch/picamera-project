import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pytransform.rotations import *
from pytransform.transformations import *
from pytransform.camera import *
from optimizer.python.cmaes import fmin
from skimage.filter import sobel
from skimage.color import rgb2gray
from skimage.io import imshow
from skimage.filter import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks


# Camera pose in world frame
#cam2world = transform_from(matrix_from_euler_xyz([0.35 * np.pi, 0, 0]),
#                           [0.0, -1.2, 0.63])
#cam2world = transform_from(matrix_from_euler_xyz([0.382 * np.pi, 0, 0]),
#                           [0, -1.95, 1.47])
e_xyz = np.array([0.12, -0.04, -0.02]) * np.pi
p = np.array([-0.83, -1.22, 2.06])
cam2world = transform_from(matrix_from_euler_xyz(e_xyz), p)
# Source: http://elinux.org/Rpi_Camera_Module#Technical_Parameters
focal_length = 0.0036
sensor_size = (0.00367, 0.00274)
image_size = (640, 480)


def draw_to_image(image, points, color=[0, 255, 0]):
    for p in points:
        p = p.astype(int)
        if 0 <= p[1] < image.shape[0] and 0 <= p[0] < image.shape[1]:
            image[p[1], p[0]] = color


def line(start, end, n_points=100):
    l = np.empty((n_points, 4))
    for d in range(4):
        l[:, d] = np.linspace(start[d], end[d], n_points)
    return l


def quadrangle(p1, p2, p3, p4, n_points_per_edge=100):
    return np.vstack((line(p1, p2), line(p2, p3), line(p3, p4), line(p4, p1)))


if __name__ == "__main__":
    filename = "data/1417558942.jpg"
    if len(sys.argv) > 1:
        filename = sys.argv[-1]
    im = np.array(Image.open(filename))

    projection_args = {"sensor_size": sensor_size, "image_size": image_size,
                       "focal_length": focal_length}

    P_corners = np.array([[ 0.000, 0.0, 0, 1],
                          [-0.100, 0.6, 0, 1],
                          [-0.880, 0.6, 0, 1],
                          [-1.315, 0.6, 0, 1],])
    P_image_corners = np.array([[464, 319],
                                [421, 198],
                                [245, 189],
                                [144, 183]])

    def objective(params):
        e_xyz = params[:3]
        p = params[3:6]
        kappa = params[-1]
        cam2world = transform_from(matrix_from_euler_xyz(e_xyz), p)
        P_image = world2image(P_corners, cam2world, kappa=kappa,
                              **projection_args)
        error = np.linalg.norm(P_image - P_image_corners) ** 2
        return error

    bounds = np.array([[-2 * np.pi, 2 * np.pi],
                       [-2 * np.pi, 2 * np.pi],
                       [-2 * np.pi, 2 * np.pi],
                       [-0.77, -0.9],
                       [-1.15, -1.3],
                       [2.0, 2.2],
                       [0, 0.1]])
    covariance = np.array([np.pi / 2, np.pi / 2, np.pi / 2, 0.1, 0.1, 0.1, 0.01])
    r = fmin(objective, "ipop", x0=np.hstack((e_xyz, p, [0])), maxfun=2000,
             log_to_stdout=True, covariance=covariance, bounds=bounds,
             random_state=0)
    print "Parameters: ", np.round(r[0], 3), ", SSE: ", np.round(r[1], 2)
    params = r[0]
    cam2world = transform_from(matrix_from_euler_xyz(params[:3]), params[3:6])
    image_corners = world2image(P_corners, cam2world, kappa=params[-1],
                                **projection_args)

    world_grid = make_world_grid(n_points_per_line=101)
    image_grid = world2image(world_grid, cam2world, kappa=params[-1],
                             **projection_args)

    plt.figure()
    img = rgb2gray(im)
    sobel_image = sobel(img)
    thresh = threshold_otsu(sobel_image)
    imt = sobel_image > thresh
    imshow(imt)

    hspace, angles, dists = hough_line(imt)
    hspace, angles, dists = hough_line_peaks(
        hspace, angles, dists, threshold=150.0)
    rows, cols = imt.shape
    for _, angle, dist in zip(hspace, angles, dists):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
        plt.gca().plot((0, cols), (y0, y1), "r")
    plt.gca().axis((0, cols, rows, 0))
    plt.gca().imshow(img, cmap=plt.cm.gray)

    plt.figure()
    draw_to_image(im, P_image_corners, color=[0, 0, 255])
    draw_to_image(im, image_grid)
    draw_to_image(im, image_corners)
    plt.imshow(im)

    plt.show()
