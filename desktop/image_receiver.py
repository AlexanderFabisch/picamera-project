import zmq
import numpy as np
from network import recv_array


context = zmq.Context()
sock = context.socket(zmq.REQ)
sock.connect("tcp://192.168.178.27:5678")

while True:
    sock.send("")
    image = recv_array(sock)

    from skimage import io, filter, color
    gray = color.rgb2gray(image)
    edges = filter.sobel(gray)
    io.imshow(edges)
    io.show()
