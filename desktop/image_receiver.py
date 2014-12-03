import time
import zmq
from network import recv_array
from PIL import Image


if __name__ == "__main__":
    context = zmq.Context()
    sock = context.socket(zmq.REQ)
    sock.connect("tcp://192.168.178.27:5678")

    sock.send("")
    a = recv_array(sock)
    im = Image.fromarray(a)
    #gray = color.rgb2gray(im)
    timestamp = "%d" % time.time()
    im.save("data/%s.jpg" % timestamp)
