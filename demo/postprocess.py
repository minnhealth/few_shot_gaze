import argparse
import cv2
import numpy as np
import pickle
import torch
import os
import csv

from person_calibration import fine_tune_from_pkl
from models import DTED
from frame_processor import frame_processer
from monitor import monitor_windows

parser = argparse.ArgumentParser(description="FAZE: post-process")
parser.add_argument(
    "--id",
    help="The ID of the participant."
)
parser.add_argument(
    "--data-dir",
    help="The path of the participant's session data directory."
)
parser.add_argument(
    "--camera", "-c",
    type=int, default=2,
    help="Webcam index for live mode (default: 2)."
)
args = parser.parse_args()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load empty gaze model
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

# Frame processor and monitor setup
cam_idx = args.camera

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
calib_path = f"calib_cam{cam_idx}.pkl"
cam_calib = pickle.load(open(calib_path, "rb"))

frame_processor = frame_processer(cam_calib)
mon = monitor_windows()

# Load weights if available
ted_parameters_path = 'demo_weights/weights_ted.pth.tar'
k = 9

assert os.path.isfile(ted_parameters_path)
print('> Loading: %s' % ted_parameters_path)
ted_weights = torch.load(ted_parameters_path)
if next(iter(ted_weights.keys())).startswith('module.'):
    ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])

maml_parameters_path = 'demo_weights/weights_maml'
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

# Run fine-tuning
calib_data_path = os.path.join(args.data_dir, "FAZE", "calib_data.pkl")
gaze_network = fine_tune_from_pkl(
    args.id, calib_data_path, frame_processor, mon, device,
    gaze_network, k=9, steps=1000, lr=1e-5, show=False
)

csv_path = os.path.join(args.data_dir, f"FAZE.csv")
csv_file = open(csv_path, mode="w", newline="")
fieldnames = ["timestamp", "x", "y"]
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()

# Run main loop
video_input_path = os.path.join(args.data_dir, "video", "video_input.mp4")
cap = cv2.VideoCapture(video_input_path)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"[INFO] Total frames: {int(total_frames)}")

# TODO: Remove once done testing with video
# video_output_path = os.path.join(args.data_dir, "FAZE", "output.mp4")
# width, height = mon.w_pixels, mon.h_pixels
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# videowriter = cv2.VideoWriter(video_output_path, fourcc, 60, (width, height))


last_log = 0.0
ret, frame = cap.read()

while ret:
    data = frame_processor.process_one(args.id, frame, mon, device, gaze_network)

    # Write to CSV
    # disp = np.zeros((mon.h_pixels, mon.w_pixels, 3), dtype=np.uint8) # TODO: Remove once done testing with video
    preds = data.get("pixel_pred", [])

    if preds:
        x_hat, y_hat = preds[0]
        x_hat = int(x_hat.item())
        y_hat = int(y_hat.item())
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        writer.writerow({
            "timestamp": int(timestamp),
            "x": x_hat,
            "y": y_hat
        })
        # cv2.circle(disp, (x_hat, y_hat), radius=30, color=(0, 255, 0), thickness=-1) # TODO: remove once done testing with video
    else:
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        writer.writerow({
            "timestamp": int(timestamp),
            "x": "",
            "y": ""
        })

    # log progress every second
    # videowriter.write(disp) # TODO: remove once done testing with video
    now_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if now_s - last_log >= 1.0:
        pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
        print(f"[PROGRESS] {now_s:.1f}s  frame {int(frame_idx)}/{int(total_frames)} ({pct:.1f}%)")
        last_log = now_s

    ret, frame = cap.read()

# Cleanup
cap.release()
# videowriter.release() # TODO: remove once done testing with video
csv_file.close()

files_to_delete = [
    f"{args.id}_gaze_network.pth.tar",
    f"{args.id}_calib.avi",
    f"{args.id}_calib_target.pkl"
]
for file_path in files_to_delete:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print(f"File not found, skipping: {file_path}")

print("[INFO] Done.")