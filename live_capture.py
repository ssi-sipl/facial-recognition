import tkinter as tk
from tkinter import messagebox
from picamera2 import Picamera2
import time
import cv2
import numpy as np
import os
import datetime

class LiveFeedApp:
    def __init__(self, master):
        self.master = master
        master.title("Live Feed")

        # Initialize camera
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_still_configuration())
        self.picam2.start()

        # Create a label to display the image
        self.label = tk.Label(master)
        self.label.pack()

        # Create a capture button
        self.capture_btn = tk.Button(master, text="Capture Image", command=self.capture_image)
        self.capture_btn.pack()

        # Start live feed
        self.update_feed()

    def update_feed(self):
        # Capture a frame
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))  # Resize for display
        self.photo = tk.PhotoImage(master=self.master, width=640, height=480, image=self.create_image(frame))
        self.label.configure(image=self.photo)
        self.label.image = self.photo  # Keep a reference
        self.master.after(10, self.update_feed)  # Update frame every 10 ms

    def create_image(self, frame):
        # Convert the NumPy array to a PhotoImage
        return cv2.imencode('.png', frame)[1].tobytes()

    def capture_image(self):
        # Create a unique filename based on the current time
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.jpg"
        self.picam2.capture_file(filename)
        messagebox.showinfo("Capture", f"Image saved as: {filename}")

    def on_closing(self):
        self.picam2.stop()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveFeedApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
