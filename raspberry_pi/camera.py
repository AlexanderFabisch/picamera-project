import picamera
from time import sleep
import datetime


CAPTURE_TABLE = "8:00,9:00,10:00,11:00,12:00,13:00,14:00,15:00,16:00,17:00,18:00,19:00"


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
    camera.vflip = False
    camera.crop = (0.0, 0.0, 1.0, 1.0)


def capture_loop(camera):
    while True:
        for t in CAPTURE_TABLE.split(","):
            hour, minute = map(int, t.split(":"))
            sleep_until(hour, minute, 1)
            now = datetime.datetime.now()
            filename = "image_%d_%d_%d_%d_%d.jpg" % (now.year, now.month, now.day, hour, minute)
            camera.capture(filename)


def sleep_until(hour, minute, verbose=0):
    now = datetime.datetime.now()
    future = datetime.datetime(now.day, now.month, now.day, hour, minute)
    delta = (future - now).seconds
    if verbose:
        print("Sleeping for %d seconds" % delta)
    sleep(delta)


if __name__ == "__main__":
    camera = picamera.PiCamera()

    set_defaults(camera)
    capture_loop(camera)

    #camera.start_recording("video.h264")
    #sleep(5)
    #camera.stop_recording()
