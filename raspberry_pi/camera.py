#!/usr/bin/python
import io
import datetime
import itertools
import picamera
import zmq
import numpy as np
from time import sleep
from daemon import runner
from PIL import Image
from network import send_array


def set_defaults(camera):
    camera.resolution = (640, 480)
    camera.sharpness = 0
    camera.contrast = 0
    camera.brightness = 50
    camera.saturation = 0
    camera.ISO = 0
    camera.video_stabilization = False
    camera.exposure_compensation = 0
    camera.exposure_mode = "auto"
    camera.meter_mode = "average"
    camera.awb_mode = "auto"
    camera.image_effect = "none"
    camera.color_effects = None
    camera.rotation = 0
    camera.hflip = True
    camera.vflip = True
    camera.crop = (0.0, 0.0, 1.0, 1.0)


def capture_as_ndarray(camera):
    stream = io.BytesIO()
    camera.capture(stream, format="jpeg")
    stream.seek(0)
    image = Image.open(stream)
    return np.asarray(image)


def capture_loop(camera):
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.bind("tcp://192.168.178.27:5678")
    while True:
        message = sock.recv()
        image = capture_as_ndarray(camera)
        send_array(sock, image)
        sleep(1)


def sleep_until(hour, minute, verbose=0):
    now = datetime.datetime.now()
    future = datetime.datetime(now.day, now.month, now.day, hour, minute)
    delta = (future - now).seconds
    if verbose:
        print("Sleeping for %d seconds" % delta)
    sleep(delta)


class App():
    def __init__(self, training=True, logging=False):
        self.training = training

        self.stdin_path = '/dev/null'
        if logging:
            self.stdout_path = '/home/pi/cam.stdout.log'
            self.stderr_path = '/home/pi/cam.stderr.log'
        else:
            self.stdout_path = '/dev/tty'
            self.stderr_path = '/dev/tty'
        self.pidfile_path =  '/tmp/cam.pid'
        self.pidfile_timeout = 5

    def run(self):
        camera = picamera.PiCamera()
        set_defaults(camera)
        capture_loop(camera)


app = App()
daemon_runner = runner.DaemonRunner(app)
daemon_runner.do_action()
