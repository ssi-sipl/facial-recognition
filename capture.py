from picamera import PiCamera
import time

camera = PiCamera()
camera.start_preview()


time.sleep(5)  # Camera warm-up time
camera.capture('./image.jpg')
camera.stop_preview()