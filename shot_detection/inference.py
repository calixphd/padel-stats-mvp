import os
import sys
import cv2
import tqdm
import logging
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

LOGGER = logging.getLogger(__name__)
MODEL_PATH = "/notebooks/shot_detection/runs/pose/train/weights/best.pt"
OUTPUTS_PATH = os.path.join("outputs", str(datetime.now()).replace(":", "_"))
FPS = 25
VIDEO_FORMATS = ["mp4", "mkv", "avi"]
BOX_THICKNESS: int = 2
TEXT_THICKNESS: int = 2
FONT: int = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: float = 0.4
BOX_COLOR = (255, 2200, 100)
TEXT_COLOR = (0, 0, 0)
CONF_THRESHOLD = 0.35


if __name__ == "__main__":
    # INSTALL DEPENDENCIES: pip install ultralytics
    # HOW TO RUN: python inference.py ../<path/to/video>
    try:
        input_file = sys.argv[1]
        print(input_file)
    except IndexError:
        LOGGER.error(
            ("This script expects a path to the input video file "
             "to run inference on")
        )
        sys.exit(1)
        
    
    if input_file.split(".")[-1] not in VIDEO_FORMATS:
        LOGGER.error(
            (f"expects input file {input_file} to be one of the "
            f"following formats: {VIDEO_FORMATS}")
        )
        sys.exit(1)
    
    if not os.path.isfile(input_file):
        LOGGER.error(f"{input_file} does not exist (no such path found)")
        sys.exit(1)
        
    summary = {
        "frame": [], 
        "confidence": [], 
        "class": [], 
        "X": [], 
        "Y": [], 
        "H": [], 
        "W": []
    }
    
    model = YOLO(MODEL_PATH)
       
    results = model(input_file, stream=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vwriter = None
    
    os.makedirs(OUTPUTS_PATH, exist_ok=False)

    # Process results list
    for (idx, result) in tqdm.tqdm(enumerate(results)):
        if(vwriter is None):
            orig_shape = result.orig_shape
            vwriter = cv2.VideoWriter(
                os.path.join(OUTPUTS_PATH, f"{os.path.basename(input_file)}_video.mp4"),
                fourcc=fourcc, 
                fps=FPS,
                frameSize=(orig_shape[1], orig_shape[0])
            )
        img = result.orig_img
        boxes = result.boxes
        
#         if boxes.conf.shape[0] == 0:
#             vwriter.write(img)
#             continue
                
        for i in range(boxes.conf.shape[0]):
            conf = boxes.conf[i].item()
            if conf < CONF_THRESHOLD:
                continue
            cls = boxes.cls[i].int().item()
            class_name = result.names[cls]
            x, y, w, h = boxes.xywh[i].cpu().round().int().numpy()
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().round().int().numpy()
            
            summary["frame"].append(idx)
            summary["confidence"].append(conf)
            summary["class"].append(class_name)
            summary["X"].append(x)
            summary["Y"].append(y)
            summary["W"].append(w)
            summary["H"].append(h)
            
            text = f"({class_name} {conf :.2f})"
            text_size = cv2.getTextSize(text, FONT, FONT_SCALE, TEXT_THICKNESS)[0]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
            img = cv2.rectangle(img, (x1, y1-text_size[1]-4), (x1+text_size[0]+2, y1), BOX_COLOR, cv2.FILLED)
            img = cv2.putText(
                img,
                text=text, 
                org=(x1, y1-2),
                color=TEXT_COLOR, 
                fontFace=FONT,
                fontScale=FONT_SCALE, 
                thickness=TEXT_THICKNESS
            )
        
        keypoints = result.keypoints.data
        keypoints = keypoints.reshape(-1, keypoints.shape[-1]) 
        keypoints = keypoints.round().int().cpu().numpy()
        for i in range(keypoints.shape[0]):
            if keypoints[i][2] == 0:
                color = (255, 255, 255)
            elif keypoints[i][2] == 1:
                color = (255, 255, 100)
            img = cv2.circle(img, keypoints[i][:2], 3, color=color, thickness=-1)
            
        # vwriter.write(img)
        
    if vwriter is not None:
        vwriter.release()
    # pd.DataFrame.from_dict(summary).to_csv(os.path.join(OUTPUTS_PATH, "output.csv"), index=False)
    output_csv_path = os.path.join(OUTPUTS_PATH, f"{os.path.basename(input_file)}_output.csv")
    pd.DataFrame.from_dict(summary).to_csv(output_csv_path, index=False)
