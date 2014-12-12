import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pytransform.rotations import *
from pytransform.transformations import *
from pytransform.camera import *
from optimizer.python.cmaes import fmin
from skimage.io import imshow
from skimage.color import rgb2gray
from skimage.filter import sobel, threshold_otsu
from skimage.transform import hough_line, hough_line_peaks


# Camera pose in world frame
e_xyz = np.array([0.12, 1.0, 0.0]) * np.pi
p = np.array([-0.83, -1.1, 2.1])
cam2world = transform_from(matrix_from_euler_xyz(e_xyz), p)
# Source: http://elinux.org/Rpi_Camera_Module#Technical_Parameters
focal_length = 0.0036
sensor_size = (0.00367, 0.00274)
image_size = (640, 480)


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


def line(start, end, n_points=100):
    l = np.empty((n_points, 4))
    for d in range(4):
        l[:, d] = np.linspace(start[d], end[d], n_points)
    return l


def quadrangle(p1, p2, p3, p4, n_points_per_edge=100):
    return np.vstack((line(p1, p2), line(p2, p3), line(p3, p4), line(p4, p1)))


def optimize_transform(projection_args):
    if os.path.exists("transform.npy"):
        return np.loadtxt("transform.npy")

    def objective(params):
        e_xyz = params[:3]
        p = params[3:6]
        kappa = params[-1]
        cam2world = transform_from(matrix_from_euler_xyz(e_xyz), p)
        P_image = world2image(P_corners, cam2world, kappa=kappa,
                              **projection_args)
        error = np.linalg.norm(P_image - P_image_corners) ** 2
        return error

    bounds = np.array([[0, 2 * np.pi],
                       [0, 2 * np.pi],
                       [0, 2 * np.pi],
                       [-0.81, -0.85],
                       [-1.08, -1.12],
                       [2.08, 2.12],
                       [0, 0.05]])
    covariance = np.array([np.pi / 2, np.pi / 2, np.pi / 2, 0.1, 0.1, 0.1, 0.01])
    r = fmin(objective, "ipop", x0=np.hstack((e_xyz, p, [0])), maxfun=2000,
             log_to_stdout=True, covariance=covariance, bounds=bounds,
             random_state=0)
    params = r[0]

    np.savetxt("transform.npy", params)
    return params


def line_y(x, a, d):
    return np.array([x, (d - x * np.cos(a)) / np.sin(a)]).astype(int)


if __name__ == "__main__":
    filename = "data/1418332576.jpg"
    if len(sys.argv) > 1:
        filename = sys.argv[-1]
    im = np.array(Image.open(filename))
    rows, cols = im.shape[:2]

    projection_args = {"sensor_size": sensor_size, "image_size": image_size,
                       "focal_length": focal_length}

    P_corners = np.array([[ 0.000, 0.0, 0, 1],
                          [-0.100, 0.6, 0, 1],
                          [-0.880, 0.6, 0, 1],
                          [-1.315, 0.6, 0, 1],])
    P_image_corners = np.array([[420, 240],
                                [374, 120],
                                [194, 114],
                                [81, 115]])

    params = optimize_transform(projection_args)
    print("Parameters: %s" % np.round(params, 3))
    cam2world = transform_from(matrix_from_euler_xyz(params[:3]), params[3:6])
    image_corners = world2image(P_corners, cam2world, kappa=params[-1],
                                **projection_args)

    world_grid = make_world_grid(n_points_per_line=101)
    image_grid = world2image(world_grid, cam2world, kappa=params[-1],
                             **projection_args)

    plt.figure(figsize=(20, 10))

    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    ax = plt.subplot(231)
    plt.setp(ax, xticks=(), yticks=())
    ax.imshow(im)

    img = rgb2gray(im)
    sobel_image = sobel(img)
    thresh = threshold_otsu(sobel_image)
    im_tresh = sobel_image > thresh

    ax = plt.subplot(232)
    plt.setp(ax, xticks=(), yticks=())
    ax.imshow(im_tresh, cmap=plt.cm.gray)

    hough_accumulator, angles, dists = hough_line(im_tresh)
    hspace, angles, dists = hough_line_peaks(
        hough_accumulator, angles, dists, threshold=150.0)
    Y = np.empty((len(hspace), 2))
    for i, angle, dist in zip(range(len(hspace)), angles, dists):
        Y[i, 0] = line_y(0, angle, dist)[1]
        Y[i, 1] = line_y(cols, angle, dist)[1]
    for y in Y:
        ax.plot((0, cols), y, "r")
    ax.axis((0, cols, rows, 0))
    plt.setp(ax, xticks=(), yticks=())

    im_edges = np.copy(im)
    # raw pixels in vicinity of lines)
    thresh_px = np.nonzero(im_tresh)
    for i in range(len(hspace)):
        for px in zip(thresh_px[1], thresh_px[0]):
            px1 = line_y(px[0] - 1, angles[i], dists[i])
            px2 = line_y(px[0], angles[i], dists[i])
            px3 = line_y(px[0] + 1, angles[i], dists[i])
            if px1[0] > px3[0]:
                tmp = px3
                px3 = px1
                px1 = tmp
            if px1[1] - 10 <= px[1] <= px3[1] + 10:
                draw_to_image(im_edges, np.atleast_2d(px), [255, 255, 0])

    ax = plt.subplot(233)
    ax.imshow(im_edges)

    im_frames = np.copy(im)
    draw_to_image(im_frames, P_image_corners, color=[0, 0, 255], thick=True)
    draw_to_image(im_frames, image_grid)
    draw_to_image(im_frames, image_corners, thick=True)

    ax = plt.subplot(234)
    plt.setp(ax, xticks=(), yticks=())
    ax.imshow(im_frames)

    ax = plt.subplot(235, projection="3d")
    plot_transform(ax)
    plot_transform(ax, A2B=cam2world)
    ax.scatter(world_grid[:, 0], world_grid[:, 1], world_grid[:, 2], s=1, c="g")
    ax.scatter(P_corners[:, 0], P_corners[:, 1], P_corners[:, 2], c="g")
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2.5, 1.5))
    ax.set_zlim((-0.2, 2.8))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # TODO project edges from world to image and compare to lines

    plt.show()
