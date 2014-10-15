#!/usr/bin/python
import picamera
from time import sleep
import datetime
import itertools
from daemon import runner


CAPTURE_TABLE = ",".join(map(lambda t: "%d:%d" % t, itertools.product(range(8, 19), range(0, 60, 1))))


def set_defaults(camera):
    camera.resolution = (2560, 1536)
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
    camera.hflip = False
    camera.vflip = True
    camera.crop = (0.0, 0.0, 1.0, 1.0)


def capture_loop(camera):
    i = 0
    while True:
        for t in CAPTURE_TABLE.split(","):
            hour, minute = map(int, t.split(":"))
            sleep_until(hour, minute, 1)
            now = datetime.datetime.now()
            filename = "/home/pi/image_%03d.jpg" % i
            camera.capture(filename)
            i += 1


def sleep_until(hour, minute, verbose=0):
    now = datetime.datetime.now()
    future = datetime.datetime(now.day, now.month, now.day, hour, minute)
    delta = (future - now).seconds
    if verbose:
        print(future)
        print("Sleeping for %d seconds" % delta)
    sleep(delta)


class App():
    def __init__(self):
        self.stdin_path = '/dev/null'
        self.stdout_path = '/home/pi/cam.stdout.log'
        self.stderr_path = '/home/pi/cam.stderr.log'
        self.pidfile_path =  '/tmp/cam.pid'
        self.pidfile_timeout = 5

    def run(self):
        camera = picamera.PiCamera()
        set_defaults(camera)
        capture_loop(camera)

        #camera.capture("image.jpg")
        #camera.start_recording("video.h264")
        #sleep(5)
        #camera.stop_recording()


app = App()
daemon_runner = runner.DaemonRunner(app)
daemon_runner.do_action()
