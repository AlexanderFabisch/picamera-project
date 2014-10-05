import picamera
from time import sleep
import datetime


def capture(camera):
    now = datetime.datetime.now()
    for i in range(10):
        filename = "image_%d_%d_%d_%d.jpg" % (i, now.year, now.month, now.day)
        camera.capture(filename)


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


if __name__ == "__main__":
    camera = picamera.PiCamera()

    set_defaults(camera)
    capture(camera)

    #camera.start_recording("video.h264")
    #sleep(5)
    #camera.stop_recording()
