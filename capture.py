from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()
time.sleep(3)  # Give time for auto-exposure adjustments

picam2.capture_file("image.jpg")
print("Image captured successfully!")

