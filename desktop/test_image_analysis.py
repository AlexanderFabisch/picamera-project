import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from image_processing import *
from image_debugging import *
from config import *
import numpy as np
from pytransform.rotations import *
from pytransform.transformations import *
from pytransform.camera import *


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise Exception("No image specified")

    image = load_image(sys.argv[1])
    rows, cols = image.shape[:2]

    door_closed, W2I, res = check_door(image, Pw_corners, Pi_corners,
                                       [Pw_door_lo, Pw_door_hi], verbose=1)

    # Grid that we display for debugging purposes
    Pw_grid = make_world_grid(n_points_per_line=101)

    # Transform points to image coordinates
    Pi_corners_proj = W2I(Pw_corners)
    Pi_grid = W2I(Pw_grid)

    if door_closed:
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
    ax.imshow(res["image_sobel"], cmap=plt.cm.gray)

    ax = plt.subplot(233)
    plt.setp(ax, xticks=(), yticks=())
    ax.imshow(res["image_edges"], cmap=plt.cm.gray)
    lines = line_points(cols, res["lines"][0], res["lines"][1])
    for l in lines:
        ax.plot((l[0, 0], l[1, 0]), (l[0, 1], l[1, 1]), "r")
    ax.axis((0, cols, rows, 0))
    plt.setp(ax, xticks=(), yticks=())

    ax = plt.subplot(234)
    image_edge_pixels = np.copy(image)
    draw_to_image(image_edge_pixels, res["Pi_line_points"], [255, 255, 0])
    ax.imshow(image_edge_pixels)

    ax = plt.subplot(235)
    plt.setp(ax, xticks=(), yticks=())
    image_frames = np.copy(image)
    draw_to_image(image_frames, Pi_corners, color=[0, 0, 255], thick=True)
    draw_to_image(image_frames, Pi_grid, color=[255, 255, 0])
    draw_to_image(image_frames, Pi_corners_proj, thick=True)
    if "door_edges_in_image" in res:
        draw_to_image(image_frames, res["door_edges_in_image"][0], thick=True)
        draw_to_image(image_frames, res["door_edges_in_image"][1], thick=True)
    ax.imshow(image_frames)

    ax = plt.subplot(236, projection="3d")
    plot_transform(ax)
    plot_transform(ax, A2B=res["cam2world"])
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
