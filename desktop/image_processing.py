import os
import numpy as np
from PIL import Image
from functools import partial
from skimage.color import rgb2gray
from skimage.filter import sobel, threshold_otsu
from skimage.transform import hough_line, hough_line_peaks
from scipy.spatial.distance import cdist
from scipy.optimize import leastsq
from pytransform.camera import world2image
from pytransform.rotations import matrix_from_euler_xyz
from pytransform.transformations import transform_from
from config import camera_params


def load_image(filename):
    return np.array(Image.open(filename))


def detect_edges(image):
    # 1. convert to grayscale image
    image_gray = rgb2gray(image)
    # 2. convolve with Sobel filter
    image_sobel = sobel(image_gray)
    # 3. compute binary edge image with threshold from Otsu's method
    image_edges = image_sobel > threshold_otsu(image_sobel)
    return image_sobel, image_edges


def optimize_transform(projection_args, Pw_corners, Pi_corners):
    """Optimize camera transformation."""
    if os.path.exists("transform.npy"):
        return np.loadtxt("transform.npy")

    # Camera pose in world frame
    e_xyz = np.array([0.0, 1.0, 0.0]) * np.pi
    p = np.array([-0.83, -1.1, 2.1])
    kappa = 0.0

    initial_params = np.hstack((e_xyz, [kappa]))
    bounds = np.array([[0, 2 * np.pi], [0, 2 * np.pi], [0, 2 * np.pi], [0, 0.05]])
    scaling = np.array([np.pi / 2, np.pi / 2, np.pi / 2, 0.01])

    def objective(params):
        params = np.clip(params, bounds[:, 0], bounds[:, 1])
        e_xyz = params[:3]
        kappa = np.clip(params[-1], 0.0, 0.1)
        cam2world = transform_from(matrix_from_euler_xyz(e_xyz), p)
        Pi_projected = world2image(Pw_corners, cam2world, kappa=kappa,
                                   **projection_args)
        return np.sum((Pi_projected - Pi_corners) ** 2, axis=1)

    r = leastsq(objective, initial_params, diag=scaling)
    params = np.clip(r[0], bounds[:, 0], bounds[:, 1])
    params = np.hstack((params[:3], p, [params[-1]]))

    np.savetxt("transform.npy", params)
    return params


def line_point(x, a, d):
    """Compute coordinates of a point on the line defined by angle and dist."""
    return np.array([x, (d - x * np.cos(a)) / np.sin(a)]).astype(int)


def check_edge_is_on_line(image_edges, angles, dists):
    """Return edge pixels that are in the vicinity of a line."""
    Pi_line_points = []
    thresh_px = np.nonzero(image_edges)
    for i in range(len(angles)):
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


def check_door(image, Pw_corners, Pi_corners, door_edges,
               required_matching_ratio=0.7, verbose=0):
    """Check if door is closed."""
    image_sobel, image_edges = detect_edges(image)

    # Detect lines with Hough transform
    hough_accumulator, angles, dists = hough_line(image_edges)
    hspace, angles, dists = hough_line_peaks(
        hough_accumulator, angles, dists, threshold=150.0)

    # Estimate camera transformation by minimizing the distance between
    # calibration points
    params = optimize_transform(camera_params, Pw_corners, Pi_corners)
    if verbose >= 1:
        print("Parameters: %s" % np.round(params, 3))
    cam2world = transform_from(matrix_from_euler_xyz(params[:3]), params[3:6])
    kappa = params[-1]

    W2I = partial(world2image, cam2world=cam2world, kappa=kappa,
                  **camera_params)

    # Get edge pixels in vicinity of lines
    Pi_line_points = check_edge_is_on_line(image_edges, angles, dists)

    # Check how good the edges of the door projected to the image match
    # detected edge pixels that correspond to lines
    matchings = [check_line_is_edge(edge, Pi_line_points, cam2world, kappa,
                                    camera_params) for edge in door_edges]
    door_edges_in_image = [m[0] for m in matchings]
    ratios = np.array([m[1] for m in matchings])

    if verbose >= 1:
        print(("Matching ratios: " + ", ".join(["%.2f"] * len(ratios)))
              % tuple(100 * ratios))

    door_closed = np.any(ratios > required_matching_ratio)

    return door_closed, W2I, {"cam2world": cam2world,
                              "Pi_line_points": Pi_line_points,
                              "door_edges_in_image": door_edges_in_image,
                              "image_sobel": image_sobel,
                              "image_edges": image_edges,
                              "lines": (angles, dists)}
