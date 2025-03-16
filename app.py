import os
import time
import csv
import torch
import cv2
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Process, Queue
import gradio as gr
from multiprocessing import get_context
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from ball2.yolov99.models.common import DetectMultiBackend
from ball2.yolov99.utils.general import (
    check_file, check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
)
from ball2.yolov99.utils.torch_utils import select_device
from ball2.yolov99.utils.dataloaders import LoadImages
from LATransformer.LATransformer.model import LATransformerTest
import timm
from torchvision import transforms
from court.voc.court import process_first_frame, threshold_mask, clean_mask, get_net, clean_fullmask, get_minmax

########################## Utility Functions ##########################

def maskpc(mask, target_class_id):
    binary_mask = (mask == target_class_id)
    return binary_mask

def get_class(x_pixel, y_pixel, max_y, min_y, max_x, min_x): 
    if  min_y[0]-1 <= y_pixel < max_y[0]:  
        class_id = "base area2"
    elif min_y[1] <= y_pixel < max_y[1]:  
        class_id = "nomans land 2"
    elif min_y[2] <= y_pixel < max_y[2]:  
        class_id = "net area2"
    elif min_y[3] <= y_pixel < max_y[3]:  
        class_id = "net area1"
    elif min_y[4] <= y_pixel < max_y[4]: 
        class_id = "nomansland1"
    elif min_y[5] <= y_pixel < max_y[5]+1:
        class_id = "nomansland1"
    else:
        class_id = "out of court"
    return class_id

def calculate_position(x_pixel, y_pixel, max_y, min_y, max_x, min_x):
    if min_y[0] <= y_pixel < max_y[0]:  # base area2
        x_meter = round((x_pixel - min_x[0]) / ((max_x[0] - min_x[0]) / 10), 2)
        y_meter = round((y_pixel - min_y[0]) * 3 / (max_y[0] - min_y[0]), 2)
    elif min_y[1] <= y_pixel < max_y[1]:  # nomans land 2
        x_meter = round((x_pixel - min_x[1]) / ((max_x[1] - min_x[1]) / 10), 2)
        y_meter = round((y_pixel - min_y[1]) * 3.5 / (max_y[1] - min_y[1]) + 3, 2)
    elif min_y[2] <= y_pixel < max_y[2]:  # net area2
        x_meter = round((x_pixel - min_x[2]) / ((max_x[2] - min_x[2]) / 10), 2)
        y_meter = round((y_pixel - min_y[2]) * 3.5 / (max_y[2] - min_y[2]) + 6.5, 2)
    elif min_y[3] <= y_pixel < max_y[3]:  # net area1
        x_meter = round((x_pixel - min_x[3]) / ((max_x[3] - min_x[3]) / 10), 2)
        y_meter = round((y_pixel - min_y[3]) * 3.5 / (max_y[3] - min_y[3]) + 10, 2)
    elif min_y[4] <= y_pixel < max_y[4]:  # nomansland1
        x_meter = round((x_pixel - min_x[4]) / ((max_x[4] - min_x[4]) / 10), 2)
        y_meter = round((y_pixel - min_y[4]) * 3.5 / (max_y[4] - min_y[4]) + 13.5, 2)
    elif min_y[5] <= y_pixel < max_y[5]:  # basearea1
        x_meter = round((x_pixel - min_x[5]) / ((max_x[5] - min_x[5]) / 10), 2)
        y_meter = round((y_pixel - min_y[5]) * 3 / (max_y[5] - min_y[5]) + 17, 2)
    else:
        return "out of bounds", "out of bounds"
    return x_meter, y_meter

############################## Shot Detection ##############################

def shot_detection(device, video_path, shot_csv, max_y, min_y, max_x, min_x,):
    try:
        model = YOLO("/notebooks/shot_detection/runs/pose/train/weights/best.pt").to(device)
        results = model(video_path, stream=True)

        summary = {
            "frame": [], "confidence": [], "class": [], "X": [], "Y": [], 
            "H": [], "W": [], "location": [], "X(m)": [], "Y(m)": []
        }
        for idx, result in enumerate(results):
            boxes = result.boxes
            for i in range(boxes.conf.shape[0]):
                conf = boxes.conf[i].item()
                if conf < 0.35:
                    continue
                cls = int(boxes.cls[i].item())
                x, y, w, h = boxes.xywh[i].cpu().round().int().numpy()
                x_meter, y_meter = calculate_position(x, y, max_y, min_y, max_x, min_x)
                location = get_class(x, y, max_y, min_y, max_x, min_x)
                summary["frame"].append(idx)
                summary["confidence"].append(conf)
                summary["class"].append(result.names[cls])
                summary["X"].append(x)
                summary["Y"].append(y)
                summary["W"].append(w)
                summary["H"].append(h)
                summary["location"].append(location)
                summary["X(m)"].append(x_meter)
                summary["Y(m)"].append(y_meter)

        pd.DataFrame(summary).to_csv(shot_csv, index=False)
        print(f"Shot detection completed, saved to {shot_csv}")
    except Exception as e:
        print(f"Shot detection failed: {e}")

############################## Ball Tracking ##############################

def ball_tracking(gpu_id, video_path, ball_csv, max_y, min_y, max_x, min_x, net_min, filename_without_extension, base_path):
    import numpy as np 
    try:
        torch.cuda.set_device(gpu_id)
        device = select_device(f'cuda:{gpu_id}')
        model = DetectMultiBackend('/notebooks/train25/weights/best.pt', device=device)
        stride, names = model.stride, model.names
        imgsz = check_img_size((720, 1280), s=stride)

        dataset = LoadImages(video_path, img_size=imgsz, stride=stride)
        with open(ball_csv, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['frame', 'class', 'X', 'Y', 'width', 'height', 'X (m)', 'Y (m)', 'location'])

            for path, im, im0s, vid_cap, s in dataset:
                im = torch.from_numpy(im).to(device).float() / 255.0
                if len(im.shape) == 3:
                    im = im[None]
                pred = model(im)
                pred = non_max_suppression(pred[0], 0.25, 0.45, max_det=1000)

                for i, det in enumerate(pred):
                    if len(det):
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                            x_meter, y_meter = calculate_position(xywh[0], xywh[1], max_y, min_y, max_x, min_x)
                            location = get_class(xywh[0], xywh[1], max_y, min_y, max_x, min_x)
                            csv_writer.writerow([dataset.frame, names[int(cls)], *xywh, x_meter, y_meter, location])

        print(f"Ball tracking completed, saved to {ball_csv}")

        df = pd.read_csv(ball_csv)
        # REMOVE STATIC BALLS
        print("removing static balls")

        # Filter out only 'ball' class data
        ball_data = df[df['class'] == 'ball']

        # Initialize a list to store indices of static balls to remove
        static_ball_indices = []
        dst = []
        # Loop through frames, starting from the second frame
        for i in range(1, len(ball_data)):
            prev_frame = ball_data.iloc[i - 1]['frame']
            curr_frame = ball_data.iloc[i]['frame']
            prev_frame_balls = ball_data[ball_data['frame'] == prev_frame]
            curr_frame_balls = ball_data[ball_data['frame'] == curr_frame]

            if len(curr_frame_balls) == 1:
                distance = np.sqrt(
                    (curr_frame_balls['X'].iloc[0] - prev_frame_balls['X'].iloc[0])**2 + 
                    (curr_frame_balls['Y'].iloc[0] - prev_frame_balls['Y'].iloc[0])**2
                )
                if distance < 2:
                    static_ball_indices.append(curr_frame_balls.index[0])
            else:
                for _, curr_ball in curr_frame_balls.iterrows():
                    for _, prev_ball in prev_frame_balls.iterrows():
                        if curr_ball['frame'] != prev_ball['frame']:
                            distance = np.sqrt((curr_ball['X'] - prev_ball['X'])**2 + (curr_ball['Y'] - prev_ball['Y'])**2)
                            dst.append(f"for {curr_ball['frame']} and {prev_ball['frame']} , distance is {distance}")
                            if distance < 2:
                                static_ball_indices.append(curr_ball.name)
                                break

        data_cleaned = df.drop(index=static_ball_indices)

        print("eliminating duplicates")
        df_filtered = data_cleaned
        df_filtered['frame'] = df_filtered['frame'].astype(int)
        cc = []
        processed_frames = []

        for i in range(0, df_filtered['frame'].max()):
            df_filtered_valid = df_filtered[
                (df_filtered['X (m)'] != 'out of bounds') & 
                (df_filtered['Y (m)'] != 'out of bounds')
            ]

            df_filtered_valid['Y (m)'] = df_filtered_valid['Y (m)'].astype(float)
            df_filtered_valid['X (m)'] = df_filtered_valid['X (m)'].astype(float)

            current_frame_balls = df_filtered_valid[df_filtered_valid["frame"] == i]
            if current_frame_balls.empty:
                continue

            if len(current_frame_balls) > 1:
                d = []
                for j in range(len(current_frame_balls)):
                    current_ball = current_frame_balls.iloc[j]
                    frame_found = False
                    for k in range(i-1, -1, -1):
                        previous_frame = df_filtered_valid[
                            (df_filtered_valid["frame"] == k) & 
                            (df_filtered_valid["class"] == "ball")
                        ]
                        if not previous_frame.empty:
                            previous_ball = previous_frame.iloc[0]
                            frame_found = True
                            break
                    if frame_found:
                        distance = np.sqrt(
                            (previous_ball['X (m)'] - current_ball['X (m)'])**2 + 
                            (previous_ball['Y (m)'] - current_ball['Y (m)'])**2
                        )
                        d.append(distance)
                    else:
                        d.append(np.nan)

                max_distance_index = np.argmax(d) if len(d) > 0 else None
                if max_distance_index is not None:
                    max_ball = current_frame_balls.iloc[max_distance_index]
                    cc.append(max_ball)
                    processed_frames.append(max_ball.to_dict())
            else:
                cc.append(current_frame_balls.iloc[0])
                processed_frames.append(current_frame_balls.iloc[0].to_dict())

        processed_df = pd.DataFrame(processed_frames)

        print("computing speed")
        df = processed_df
        ball_df = df[df['class'] == 'ball']
        speeds_kmh = [None]

        for i in range(1, len(ball_df)):
            x1, y1 = ball_df.iloc[i-1][['X (m)', 'Y (m)']]
            x2, y2 = ball_df.iloc[i][['X (m)', 'Y (m)']]
            frame1, frame2 = ball_df.iloc[i-1]['frame'], ball_df.iloc[i]['frame']
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            time_diff = (frame2 - frame1) / 25
            speed_mps = distance / time_diff
            speed_kmh = speed_mps * 3.6
            speeds_kmh.append(speed_kmh)

        df['Speed (km/h)'] = np.nan
        df.loc[df['class'] == 'ball', 'Speed (km/h)'] = speeds_kmh

        print("trajectoryyyyyyy")

        def check_hits_and_classify(db):
            previous_x, previous_y = None, None
            trajectory = []
            crossed_net_status = []

            for index, row in db.iterrows():
                if row['class'] == 'ball':
                    if previous_x is not None and previous_y is not None:
                        delta_x = row['X'] - previous_x
                        delta_y = row['Y'] - previous_y
                        angle_rad = np.arctan2(delta_y, delta_x)
                        angle_deg = np.degrees(angle_rad)

                        if (0 <= angle_deg <= 20) or (120 <= angle_deg <= 180):
                            trajectory.append("cross court")
                        else:
                            trajectory.append("down the line")
                    else:
                        trajectory.append(None)

                    if net_min-30 <= row['Y'] <= net_min+50:
                        crossed_net_status.append(True)
                    else:
                        crossed_net_status.append(False)

                    previous_x, previous_y = row['X'], row['Y']
                else:
                    trajectory.append(None)
                    crossed_net_status.append(None)

            db['trajectory'] = trajectory
            db['crossed_net'] = crossed_net_status
            return db

        db = check_hits_and_classify(df)
        ball_cleaned, _ = os.path.splitext(ball_csv)
        db.to_csv(os.path.join(base_path, f"{filename_without_extension}b.csv"), index=False)
        print("ball csv cleaned and saved ")
    except Exception as e:
        print(f"Ball tracking failed: {e}")

############################## Player ID Tracking ##############################

def player_id_tracking(device, video_path, player_csv, max_y, min_y, max_x, min_x, filename_without_extension, base_path):
    try:
        tracker = PersonTracker(
            reid_model_path='/notebooks/LATransformer/model/la_with_lmbd_8/net_best.pth',
            yolo_model='yolov8n.pt',
            similarity_threshold=0.6
        )
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with open(player_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Frame', 'ID', 'X', 'Y', 'Width', 'Height', "location", "X(m)", "Y(m)"])
            frame_counter = 0

            with tqdm(total=total_frames, desc="Player tracking") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_counter % 2 != 0:
                        frame_counter += 1
                        pbar.update(1)
                        continue
                    tracks = tracker.update(frame)
                    for track in tracks.values():
                        if track.missed_frames == 0:
                            x1, y1, x2, y2 = track.bbox
                            width, height = x2 - x1, y2 - y1
                            location = get_class(x1, y2, max_y, min_y, max_x, min_x)
                            x_meter, y_meter = calculate_position(x1, y2, max_y, min_y, max_x, min_x)
                            csvwriter.writerow([frame_counter, track.id, x1, y2, width, height, location, x_meter, y_meter])
                    frame_counter += 1
                    pbar.update(1)

        cap.release()
        df = pd.read_csv(player_csv)
        df = df[df["X(m)"] != "out of court"]
        first_4_ids = df['ID'].drop_duplicates().head(4).tolist()

        last_positions = {id: None for id in first_4_ids}

        for frame in sorted(df['Frame'].unique()):
            frame_df = df[df['Frame'] == frame]
            # Update last known positions
            for _, row in frame_df.iterrows():
                if row['ID'] in first_4_ids:
                    last_positions[row['ID']] = np.array([row['X'], row['Y']])

            ids_in_frame = frame_df['ID'].isin(first_4_ids)
            if not ids_in_frame.all():
                assigned_ids = set()
                for idx, (is_in, id) in enumerate(zip(ids_in_frame, frame_df['ID'])):
                    if not is_in:
                        available_ids = [
                            id_ for id_ in first_4_ids 
                            if id_ not in frame_df['ID'].values and id_ not in assigned_ids
                        ]
                        if available_ids:
                            distances = []
                            for available_id in available_ids:
                                if last_positions[available_id] is not None:
                                    current_coords = frame_df.iloc[idx][['X', 'Y']].values
                                    distance = np.linalg.norm(last_positions[available_id] - current_coords)
                                    distances.append((available_id, distance))

                            if distances:
                                new_id, _ = min(distances, key=lambda x: x[1])
                                df.loc[frame_df.index[idx], 'ID'] = new_id
                                assigned_ids.add(new_id)

        db = df[df['ID'].isin(first_4_ids)]
        csv_cleaned, _ = os.path.splitext(player_csv)
        os.path.join(base_path, f"{filename_without_extension}_s.csv")
        db.to_csv(os.path.join(base_path, f"{filename_without_extension}p.csv"), index=False)

        print(f"Player ID tracking completed, saved to {player_csv}")
    except Exception as e:
        print(f"Player ID tracking failed: {e}")

############################## Person Detector, ReID, Tracker ##############################

class PersonDetector:
    def __init__(self, model_size, device='cuda:0' if torch.cuda.device_count() > 0 else 'cpu'):
        self.model = YOLO(model_size).to(device)

    def detect(self, frame):
        results = self.model(frame, classes=0)  # Detect persons only
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if box.conf[0] > 0.5:  # Confidence threshold
                    detections.append(tuple(map(int, box.xyxy[0])))
        return detections

class ReIDModel:
    def __init__(self, model_path):
        self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
        vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
        self.model = LATransformerTest(vit_base, lmbd=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def extract_features(self, image):
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.model(img_tensor)
            fnorm = torch.norm(features, p=2, dim=1, keepdim=True) * np.sqrt(14)
            return features.div(fnorm.expand_as(features)).view(-1).cpu()

class Track:
    def __init__(self, track_id, bbox, feature):
        self.id = track_id
        self.bbox = bbox
        self.features = [feature]
        self.missed_frames = 0
    def update(self, bbox, feature):
        self.bbox = bbox
        self.features.append(feature)
        self.missed_frames = 0
    def mark_missed(self):
        self.missed_frames += 1

class PersonTracker:
    def __init__(self, reid_model_path, yolo_model="yolov8n.pt", similarity_threshold=0.6, max_missed_frames=3000):
        self.detector = PersonDetector(yolo_model)
        self.reid_model = ReIDModel(reid_model_path)
        self.tracks = {}
        self.next_id = 0
        self.similarity_threshold = similarity_threshold
        self.max_missed_frames = max_missed_frames

    def _compute_similarity(self, feat1, feat2):
        return float(torch.nn.functional.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)))

    def update(self, frame):
        detections = self.detector.detect(frame)
        features = [
            self.reid_model.extract_features(frame[y1:y2, x1:x2]) 
            for x1, y1, x2, y2 in detections 
            if frame[y1:y2, x1:x2].size > 0
        ]
        valid_detections = [d for d, f in zip(detections, features)]

        matches, unmatched = {}, set(range(len(valid_detections)))
        for track_id, track in self.tracks.items():
            if not unmatched:
                break
            track_feat = track.features[-1]
            best_match, best_similarity = None, 0
            for det_idx in unmatched:
                similarity = self._compute_similarity(track_feat, features[det_idx])
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = det_idx
            if best_match is not None:
                matches[track_id] = (best_match, features[best_match])
                unmatched.remove(best_match)

        for track_id, (det_idx, feature) in matches.items():
            self.tracks[track_id].update(valid_detections[det_idx], feature)
        for det_idx in unmatched:
            self.tracks[self.next_id] = Track(self.next_id, valid_detections[det_idx], features[det_idx])
            self.next_id += 1

        for track_id in list(self.tracks.keys()):
            if track_id not in matches:
                self.tracks[track_id].mark_missed()
                if self.tracks[track_id].missed_frames > self.max_missed_frames:
                    del self.tracks[track_id]

        return self.tracks

############################## Main Video Process ##############################

def process_vid(video_path):
    unique_id = str(time.time()).replace('.', '')
    base_path = f"results__{unique_id}"
    filename_with_extension = os.path.basename(video_path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    base_path = f"results__{filename_without_extension}_{unique_id}"
    ball_csv = os.path.join(base_path, f"{filename_without_extension}_ball.csv")
    shot_csv = os.path.join(base_path, f"{filename_without_extension}_s.csv")
    player_csv = os.path.join(base_path, f"{filename_without_extension}_player.csv")
    os.makedirs(base_path, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    pp = process_first_frame(video_path)
    cleaned_mask = threshold_mask(pp, target_class_id=3, threshold=0.5)
    cleaned_mask = clean_mask(cleaned_mask)
    net_p, net_min = get_net(cleaned_mask)
    full_cleaned_mask = clean_fullmask(pp, num_classes=7, threshold=0.5, kernel_size=3)

    def maskpc(mask,target_class_id):
        binary_mask = (mask == target_class_id)
        return binary_mask

    full = [2,7,5,4,6,1]
    cls = [7,5,4,6,1]
    i = 2
    max_y = []
    min_y = []
    max_x = []
    min_x = []

    mpc = maskpc(full_cleaned_mask, i)
    max_y_j, min_y_j, max_x_j, min_x_j = get_minmax(mpc)
    max_y.append(max_y_j)
    min_y.append(min_y_j)
    max_x.append(max_x_j)
    min_x.append(min_x_j)

    for i in range(len(cls)):
        j = cls[i]
        mpc = maskpc(full_cleaned_mask, j)
        max_y_j, min_y_j, max_x_j, min_x_j = get_minmax(mpc)
        min_y_j = max_y[-1]
        max_y.append(max_y_j)
        min_y.append(min_y_j)
        max_x.append(max_x_j)
        min_x.append(min_x_j)

    print(f" max x is   {max_x}")
    print(f" min x is   {min_x}")

    stime = time.time()
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    device = 'cuda:0' if num_gpus > 0 else 'cpu'
    ctx = get_context('spawn')

    processes = [
        ctx.Process(target=shot_detection, args=(device, video_path, shot_csv, max_y, min_y, max_x, min_x)),
        ctx.Process(target=ball_tracking, args=(device, video_path, ball_csv, max_y, min_y, max_x, min_x, net_min, filename_without_extension, base_path)),
        ctx.Process(target=player_id_tracking, args=(device, video_path, player_csv, max_y, min_y, max_x, min_x, filename_without_extension, base_path))
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    etime = time.time()
    print(f"Elapsed time: {etime - stime} seconds")
    player_cleaned_csv = os.path.join(base_path, f"{filename_without_extension}p.csv")
    ball_cleaned_csv = os.path.join(base_path, f"{filename_without_extension}b.csv")
    print(f"done processing and saved to {base_path}")

    # 1) Convert relative paths to absolute
    ball_cleaned_csv_abs = str(Path(ball_cleaned_csv).resolve())
    shot_csv_abs         = str(Path(shot_csv).resolve())
    player_cleaned_csv_abs = str(Path(player_cleaned_csv).resolve())

    # 2) Return the absolute paths
    return (ball_cleaned_csv_abs, shot_csv_abs, player_cleaned_csv_abs)

################# XLS Processing 

from tabulate import tabulate
import pandas as pd

def process_xls(xls_file):
    """
    Reads the uploaded .xls (actually CSV) file,
    calculates #, % Success, and % Winner, 
    and returns a formatted table for display.
    """
    if xls_file is None:
        return "No file uploaded."

    # 1) Treat it as CSV:
    df = pd.read_csv(xls_file.name)

    # 2) Paste your existing numeric col logic:
    numeric_cols = ["nbr", "success", "winner", "loss"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3) Identify special row types
    special_types = {
        "DISTANCE",
        "RATING",
        "BASE AREA %",
        "NML AREA %",
        "NET AREA %",
    }

    user_ids = df["user_id"].unique()

    # We'll build up a string to return
    from tabulate import tabulate
    output_str = ""

    for user_id in user_ids:
        # (Paste the rest of your logic here)
        # Filter user rows, separate shot_df vs. special_df, 
        # compute %Success, %Winner, reorder, etc.

        if pd.isna(user_id):
            user_mask = df["user_id"].isna()
        else:
            user_mask = (df["user_id"] == user_id)
        user_df = df[user_mask].copy()

        special_df = user_df[user_df["type"].isin(special_types)]
        shot_df    = user_df[~user_df["type"].isin(special_types)]

        shot_stats = shot_df[["type", "nbr", "success", "winner"]].copy()
        shot_stats.rename(columns={"type": "ShotType"}, inplace=True)

        shot_stats["% Success"] = (
            shot_stats["success"] / shot_stats["nbr"] * 100
        ).fillna(0).round(2)
        shot_stats["% Winner"] = (
            shot_stats["winner"] / shot_stats["nbr"] * 100
        ).fillna(0).round(2)

        shot_stats.rename(columns={"nbr": "#"}, inplace=True)
        shot_stats.drop(columns=["success", "winner"], inplace=True)

        desired_order = [
            "HITS",
            "DOWN THE LINE",
            "CROSS COURT",
            "FOREHAND",
            "BACKHAND",
            "FROM BASE AREA",
            "FROM NET AREA",
            "FROM NML AREA",
            "SMACH",
            "VOLLEY",
            "GROUND STROKE",
            "LOB",
            "VIBORA",
            "CHIQUITA",
            "SERVE",
        ]
        shot_stats["ShotType"] = pd.Categorical(
            shot_stats["ShotType"], 
            categories=desired_order, 
            ordered=True
        )
        shot_stats.sort_values("ShotType", inplace=True)

        distance_val = None
        rating_val   = None

        distance_rows = special_df[special_df["type"] == "DISTANCE"]
        if not distance_rows.empty:
            distance_val = distance_rows["nbr"].values[0]
        rating_rows = special_df[special_df["type"] == "RATING"]
        if not rating_rows.empty:
            rating_val = rating_rows["nbr"].values[0]

        user_label = f"User {user_id}" if pd.notna(user_id) else "User (blank)"
        output_str += f"\n=== {user_label} ===\n\n"

        if not shot_stats.empty:
            table_str = tabulate(shot_stats, headers='keys', tablefmt='pretty', showindex=False)
            output_str += table_str + "\n"
        else:
            output_str += "(No shot stats for this user)\n"

        if distance_val is not None:
            output_str += f"\nDistance RAN: {distance_val:.2f} km\n"
        if rating_val is not None:
            output_str += f"Today's RATING: {rating_val:.2f}\n"

        output_str += "\n" + "-"*50 + "\n"

    
    return f"```\n{output_str}\n```"


######### Gradio Interfaces

#  interface for video processing
iface_video = gr.Interface(
    fn=process_vid,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.File(label="Ball Detections"),
        gr.File(label="Shot Detections"),
        gr.File(label="Player Tracking")
    ],
    title="Video Processing App",
    description="Upload a video to process for ball detections, shot detections, and player tracking. Results will be available for download as CSV files."
)

# interface for XLS processing
iface_xls = gr.Interface(
    fn=process_xls,
    inputs=gr.File(label="Upload XLS File"),
    outputs=gr.Markdown(label="XLS Results"),
    title="XLS Processing",
    description="Upload an XLS file to see a pretty table of its contents (no download)."
)

demo = gr.TabbedInterface(
    [iface_video, iface_xls],
    ["Video Processing", "XLS Processing"]
)


############################## Main Execution 

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        print("Start method already set, proceeding with:", multiprocessing.get_start_method())

    demo.launch(share=True)
