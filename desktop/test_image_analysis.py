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
    """Optimize camera transformation."""
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


def line_point(x, a, d):
    """Compute coordinates of a point on the line defined by angle and dist."""
    return np.array([x, (d - x * np.cos(a)) / np.sin(a)]).astype(int)


def check_edge_is_on_line(image_edges, angles, dists):
    """Return edge pixels that are in the vicinity of a line."""
    Pi_line_points = []
    thresh_px = np.nonzero(image_edges)
    for i in range(len(hspace)):
        for px in zip(thresh_px[1], thresh_px[0]):
            px1 = line_point(px[0] - 1, angles[i], dists[i])
            px2 = line_point(px[0], angles[i], dists[i])
            px3 = line_point(px[0] + 1, angles[i], dists[i])
            if px1[0] > px3[0]:
                tmp = px3
                px3 = px1
                px1 = tmp
            if px1[1] - 10 <= px[1] <= px3[1] + 10:
                Pi_line_points.append(px)
    return np.array(Pi_line_points)


def check_line_is_edge(Pw_line, Pi_edge, cam2world, kappa, camera_params,
                       max_pixel_dist=5):
    """Check if a line defined in world frame corresponds to edge pixels."""
    Pi_line = world2image(Pw_line, cam2world, kappa=kappa, **camera_params)

    if len(Pi_line) == 0:
        return Pi_line, 0.0
    else:
        dists = cdist(Pi_line, Pi_edge)
        min_dists = dists.min(axis=1)
        n_matching = np.count_nonzero(min_dists < max_pixel_dist)
        return Pi_line, float(n_matching) / len(min_dists)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise Exception("No image specified")

    filename = sys.argv[1]
    image = np.array(Image.open(filename))
    rows, cols = image.shape[:2]

    required_matching_ratios = (0.7, 0.7)

    # Calibration points for camera parameters
    Pw_corners = np.array([[ 0.000, 0.0, 0, 1],
                           [-0.100, 0.6, 0, 1],
                           [-0.880, 0.6, 0, 1],
                           [-1.315, 0.6, 0, 1],])
    Pi_corners = np.array([[420, 240],
                           [374, 120],
                           [194, 114],
                           [81, 115]])
    # Position of visible edges from the closed door
    Pw_door_lo = make_world_line([0, 0, 0, 1], [0, -0.85, 0, 1], 51)
    Pw_door_hi = make_world_line([0, 0, 0, 1], [0, 0, 1, 1], 51)
    # Grid that we display for debugging purposes
    Pw_grid = make_world_grid(n_points_per_line=101)

    # Estimate camera transformation by minimizing the distance between
    # calibration points
    params = optimize_transform(camera_params)
    print("Parameters: %s" % np.round(params, 3))
    cam2world = transform_from(matrix_from_euler_xyz(params[:3]), params[3:6])
    kappa = params[-1]

    # Transform points to image coordinates
    Pi_corners_proj = world2image(Pw_corners, cam2world, kappa=kappa,
                                  **camera_params)
    Pi_grid = world2image(Pw_grid, cam2world, kappa=kappa,
                          **camera_params)

    # Edge detection:
    # 1. convert to grayscale image
    image_gray = rgb2gray(image)
    # 2. convolve with Sobel filter
    image_sobel = sobel(image_gray)
    # 3. compute binary edge image with threshold from Otsu's method
    image_edges = image_sobel > threshold_otsu(image_sobel)

    # Detect lines with Hough transform
    hough_accumulator, angles, dists = hough_line(image_edges)
    hspace, angles, dists = hough_line_peaks(
        hough_accumulator, angles, dists, threshold=150.0)

    # Get edge pixels in vicinity of lines
    Pi_line_points = check_edge_is_on_line(image_edges, angles, dists)

    # Check how good the edges of the door projected to the image match
    # detected edge pixels that correspond to lines
    P_door_from_world_lo, matching_ratio_lo = check_line_is_edge(
        Pw_door_lo, Pi_line_points, cam2world, kappa, camera_params)
    P_door_from_world_hi, matching_ratio_hi = check_line_is_edge(
        Pw_door_hi, Pi_line_points, cam2world, kappa, camera_params)

    print("Matching ratios: %.1f%%, %.1f%%"
          % (100 * matching_ratio_lo, 100 * matching_ratio_hi))

    if (matching_ratio_lo > required_matching_ratios[0] or
            matching_ratio_hi > required_matching_ratios[1]):
        print("The door is closed")
    else:
        print("The door is open")

    plt.figure(figsize=(20, 10))

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax = plt.subplot(231)
    plt.setp(ax, xticks=(), yticks=())
    ax.imshow(image)

    ax = plt.subplot(232)
    plt.setp(ax, xticks=(), yticks=())
    ax.imshow(image_sobel, cmap=plt.cm.gray)

    ax = plt.subplot(233)
    plt.setp(ax, xticks=(), yticks=())
    ax.imshow(image_edges, cmap=plt.cm.gray)
    lines = np.array(
        [[line_point(0, angle, dist), line_point(cols, angle, dist)]
         for i, angle, dist in zip(range(len(hspace)), angles, dists)])
    for l in lines:
        ax.plot((l[0, 0], l[1, 0]), (l[0, 1], l[1, 1]), "r")
    ax.axis((0, cols, rows, 0))
    plt.setp(ax, xticks=(), yticks=())

    ax = plt.subplot(234)
    image_edge_pixels = np.copy(image)
    draw_to_image(image_edge_pixels, Pi_line_points, [255, 255, 0])
    ax.imshow(image_edge_pixels)

    ax = plt.subplot(235)
    plt.setp(ax, xticks=(), yticks=())
    image_frames = np.copy(image)
    draw_to_image(image_frames, Pi_corners, color=[0, 0, 255], thick=True)
    draw_to_image(image_frames, Pi_grid, color=[255, 255, 0])
    draw_to_image(image_frames, Pi_corners_proj, thick=True)
    draw_to_image(image_frames, P_door_from_world_lo, thick=True)
    draw_to_image(image_frames, P_door_from_world_hi, thick=True)
    ax.imshow(image_frames)

    ax = plt.subplot(236, projection="3d")
    plot_transform(ax)
    plot_transform(ax, A2B=cam2world)
    ax.scatter(Pw_grid[:, 0], Pw_grid[:, 1], Pw_grid[:, 2], s=1, c="g")
    ax.scatter(Pw_corners[:, 0], Pw_corners[:, 1], Pw_corners[:, 2], c="g")
    ax.scatter(Pw_door_lo[:, 0], Pw_door_lo[:, 1], Pw_door_lo[:, 2], c="g")
    ax.scatter(Pw_door_hi[:, 0], Pw_door_hi[:, 1], Pw_door_hi[:, 2], c="g")
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2.5, 1.5))
    ax.set_zlim((-0.2, 2.8))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
