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
from scipy.spatial.distance import cdist


# Source: http://elinux.org/Rpi_Camera_Module#Technical_Parameters
camera_params = {"focal_length": 0.0036,
                 "sensor_size": (0.00367, 0.00274),
                 "image_size": (640, 480)}


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


def optimize_transform(projection_args):
    if os.path.exists("transform.npy"):
        return np.loadtxt("transform.npy")

    # Camera pose in world frame
    e_xyz = np.array([0.0, 1.0, 0.0]) * np.pi
    p = np.array([-0.83, -1.1, 2.1])
    kappa = 0.0
    initial_params = np.hstack((e_xyz, p, [kappa]))

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
    r = fmin(objective, "ipop", x0=initial_params, maxfun=2000,
             log_to_stdout=True, covariance=covariance, bounds=bounds,
             random_state=0)
    params = r[0]

    np.savetxt("transform.npy", params)
    return params


def line_y(x, a, d):
    return np.array([x, (d - x * np.cos(a)) / np.sin(a)]).astype(int)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise Exception("No image specified")

    filename = sys.argv[1]
    im = np.array(Image.open(filename))
    rows, cols = im.shape[:2]

    P_corners = np.array([[ 0.000, 0.0, 0, 1],
                          [-0.100, 0.6, 0, 1],
                          [-0.880, 0.6, 0, 1],
                          [-1.315, 0.6, 0, 1],])
    P_world_door_low = make_world_line([0, 0, 0, 1], [0, -0.85, 0, 1], 51)
    P_world_door_hi = make_world_line([0, 0, 0, 1], [0, 0, 1, 1], 51)
    P_world_grid = make_world_grid(n_points_per_line=101)
    P_image_corners = np.array([[420, 240],
                                [374, 120],
                                [194, 114],
                                [81, 115]])

    params = optimize_transform(camera_params)
    print("Parameters: %s" % np.round(params, 3))
    cam2world = transform_from(matrix_from_euler_xyz(params[:3]), params[3:6])
    image_corners = world2image(P_corners, cam2world, kappa=params[-1],
                                **camera_params)

    image_grid = world2image(P_world_grid, cam2world, kappa=params[-1],
                             **camera_params)

    plt.figure(figsize=(20, 10))

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax = plt.subplot(231)
    plt.setp(ax, xticks=(), yticks=())
    ax.imshow(im)

    img = rgb2gray(im)
    sobel_image = sobel(img)
    thresh = threshold_otsu(sobel_image)
    im_tresh = sobel_image > thresh

    ax = plt.subplot(232)
    plt.setp(ax, xticks=(), yticks=())
    ax.imshow(sobel_image, cmap=plt.cm.gray)

    ax = plt.subplot(233)
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
    P_line_points = []
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
                P_line_points.append(px)
    P_line_points  = np.array(P_line_points)
    draw_to_image(im_edges, P_line_points, [255, 255, 0])

    ax = plt.subplot(234)
    ax.imshow(im_edges)

    im_frames = np.copy(im)
    draw_to_image(im_frames, P_image_corners, color=[0, 0, 255], thick=True)
    draw_to_image(im_frames, image_grid, color=[255, 255, 0])
    draw_to_image(im_frames, image_corners, thick=True)
    P_door_from_world_lo = world2image(
        P_world_door_low, cam2world, kappa=params[-1], **camera_params)
    P_door_from_world_hi = world2image(
        P_world_door_hi, cam2world, kappa=params[-1], **camera_params)
    draw_to_image(im_frames, P_door_from_world_lo, thick=True)
    draw_to_image(im_frames, P_door_from_world_hi, thick=True)

    if len(P_line_points) > 0:
        dists = cdist(P_door_from_world_lo, P_line_points)
        dists.sort(axis=1)
        min_dists = dists.min(axis=1)
        edge_pixels_percentage = float(np.count_nonzero(min_dists < 5)) / len(min_dists)
        print edge_pixels_percentage
        detected_lo = 0.7 < edge_pixels_percentage

        dists = cdist(P_door_from_world_hi, P_line_points)
        dists.sort(axis=1)
        min_dists = dists.min(axis=1)
        edge_pixels_percentage = float(np.count_nonzero(min_dists < 5)) / len(min_dists)
        print edge_pixels_percentage
        detected_hi = 0.7 < edge_pixels_percentage

        door_closed = detected_lo or detected_hi
    else:
        door_closed = True

    if door_closed:
        print("The door is closed")
    else:
        print("The door is open")

    ax = plt.subplot(235)
    plt.setp(ax, xticks=(), yticks=())
    ax.imshow(im_frames)

    ax = plt.subplot(236, projection="3d")
    plot_transform(ax)
    plot_transform(ax, A2B=cam2world)
    ax.scatter(P_world_grid[:, 0], P_world_grid[:, 1], P_world_grid[:, 2], s=1, c="g")
    ax.scatter(P_corners[:, 0], P_corners[:, 1], P_corners[:, 2], c="g")
    ax.scatter(P_world_door_low[:, 0], P_world_door_low[:, 1], P_world_door_low[:, 2], c="g")
    ax.scatter(P_world_door_hi[:, 0], P_world_door_hi[:, 1], P_world_door_hi[:, 2], c="g")
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2.5, 1.5))
    ax.set_zlim((-0.2, 2.8))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
