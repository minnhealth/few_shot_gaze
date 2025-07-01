#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import time
import argparse
import cv2
import numpy as np
from os import path
from subprocess import call
import pickle
import sys
import torch
import os

import warnings
warnings.filterwarnings("ignore")

from monitor import monitor_windows
from camera import cam_calibrate
from person_calibration import collect_data, fine_tune
from frame_processor import frame_processer

# ——— Argument parsing ———
parser = argparse.ArgumentParser(description="EyeTrax Demo: live or post-process")
parser.add_argument(
    "--video", "-v",
    help="Path to input video file (post-process mode). If omitted, live webcam mode is used."
)
parser.add_argument(
    "--camera", "-c",
    type=int, default=2,
    help="Webcam index for live mode (default: 2)."
)
parser.add_argument(
    "--force-calibrate", "-f",
    action="store_true",
    help="Force a re-calibration even if a calibration for the given subject exists."
)
parser.add_argument(
    "--output", "-o",
    default="gaze_output.mp4",
    help="Output overlay video (only in post-process mode)."
)
parser.add_argument(
    "--fps", type=float, default=60.0,
    help="Frame rate for output video (post-process mode)."
)
args = parser.parse_args()

# ——— Initialize capture ———
if args.video:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {args.video}")
    post_process = True
    print(f"[INFO] Post-process mode. Reading {args.video}")
else:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")
    post_process = False
    print(f"[INFO] Live mode. Using camera {args.camera}")

# match original resolution settings :contentReference[oaicite:0]{index=0}
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ——— Camera calibration (once) ———
cam_idx = args.camera
calib_path = f"calib_cam{cam_idx}.pkl"
if os.path.exists(calib_path):
    cam_calib = pickle.load(open(calib_path, "rb"))
else:
    cam_calib = {"mtx": np.eye(3), "dist": np.zeros((1, 5))}
    print("[INFO] Calibrating camera. Press 's' to save frames, 'c' to continue.")
    cam_calibrate(cam_idx, cap, cam_calib)
    pickle.dump(cam_calib, open(calib_path, "wb"))

# ——— Monitor & Gaze network setup ———
mon = monitor_windows()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#################################
# Load gaze network
#################################
ted_parameters_path = 'demo_weights/weights_ted.pth.tar'
maml_parameters_path = 'demo_weights/weights_maml'
k = 9

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create network
sys.path.append("../src")
from models import DTED
gaze_network = DTED(
    growth_rate=32,
    z_dim_app=64,
    z_dim_gaze=2,
    z_dim_head=16,
    decoder_input_c=32,
    normalize_3d_codes=True,
    normalize_3d_codes_axis=1,
    backprop_gaze_to_encoder=False,
).to(device)

#################################

# Load T-ED weights if available
assert os.path.isfile(ted_parameters_path)
print('> Loading: %s' % ted_parameters_path)
ted_weights = torch.load(ted_parameters_path)
if torch.cuda.device_count() == 1:
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])

#####################################

# Load MAML MLP weights if available
full_maml_parameters_path = maml_parameters_path +'/%02d.pth.tar' % k
assert os.path.isfile(full_maml_parameters_path)
print('> Loading: %s' % full_maml_parameters_path)
maml_weights = torch.load(full_maml_parameters_path)
ted_weights.update({  # rename to fit
    'gaze1.weight': maml_weights['layer01.weights'],
    'gaze1.bias':   maml_weights['layer01.bias'],
    'gaze2.weight': maml_weights['layer02.weights'],
    'gaze2.bias':   maml_weights['layer02.bias'],
})
gaze_network.load_state_dict(ted_weights)

#################################
# Personalize gaze network
#################################

# ——— Personal calibration ———
frame_processor = frame_processer(cam_calib)
subject = input("Enter subject name: ")
calib_net_path = f"{subject}_gaze_network.pth.tar"
if os.path.isfile(calib_net_path) and not args.force_calibrate:
    state = torch.load(calib_net_path, map_location=device)
    gaze_network.load_state_dict(state)
    print(f"[INFO] Loaded fine-tuned network from {calib_net_path}")
else:
    data = collect_data(cap, mon, calib_points=9, rand_points=4)
    gaze_network = fine_tune(
        subject, data, frame_processor, mon, device,
        gaze_network, k=9, steps=1000, lr=1e-5, show=False
    )

# ——— Prepare output writer if needed ———
writer = None
if post_process:
    width, height = mon.w_pixels, mon.h_pixels
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"[INFO] Total frames: {int(total_frames)}")

# ——— Main loop ———
last_log = 0.0
ret, frame = cap.read()
while ret:
    data = frame_processor.process_one(subject, frame, mon, device, gaze_network)

    # render overlay
    disp = np.zeros((mon.h_pixels, mon.w_pixels, 3), dtype=np.uint8)
    preds = data.get("pixel_pred", [])
    if preds:
        x_hat, y_hat = preds[0]
        x_hat = int(x_hat.item())
        y_hat = int(y_hat.item())
        # draw a green filled circle
        cv2.circle(disp, (x_hat, y_hat), radius=30, color=(0,255,0), thickness=-1)

    if post_process:
        writer.write(disp)
        # log progress every second
        now_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if now_s - last_log >= 1.0:
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            print(f"[PROGRESS] {now_s:.1f}s  frame {int(frame_idx)}/{int(total_frames)} ({pct:.1f}%)")
            last_log = now_s
    else:
        cv2.imshow("Gaze Overlay", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    ret, frame = cap.read()

# ——— Cleanup ———
cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

print("[INFO] Done.")
