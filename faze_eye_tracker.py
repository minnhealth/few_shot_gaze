import cv2
import pickle
from FAZE.demo.person_calibration import collect_data
from FAZE.demo.monitor import monitor_windows
from threading import Thread, Event

from EyeTracker import EyeTracker

class FazeEyeTracker(EyeTracker):
    def __init__(self, camera_index=2):
        self.camera_index = camera_index
        self.cap = None
        self.running = Event()
        self.thread = None
        self._calibrated = False
        self.gaze_callback = None

    def start(self, participant_id, screen_width, screen_height):
        print("[FAZE] Starting FAZE")

    def stop(self):
        self.running.clear()
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.thread = None

    def calibrate(self, calibration_finished_callback):
        print("[FAZE] calibrate")
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        mon = monitor_windows()
        calib_data = collect_data(cap, mon, calib_points=9, rand_points=4)

        with open("./FAZE/calib_data.pkl", "wb") as f:
            pickle.dump(calib_data, f)

        print("Saved raw calibration data.")

        if calibration_finished_callback:
            calibration_finished_callback(True, "FAZE calibration finished.")

    def set_gaze_estimate_callback(self, gaze_callback):
        print("[FAZE] set_gaze_estimate_callback")
