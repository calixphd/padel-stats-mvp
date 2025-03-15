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
from ball2.yolov99.utils.general import (check_file, check_img_size, non_max_suppression, scale_boxes, xyxy2xywh)
from ball2.yolov99.utils.torch_utils import select_device
from ball2.yolov99.utils.dataloaders import LoadImages
from LATransformer.LATransformer.model import LATransformerTest
import timm
from torchvision import transforms
from court.voc.court import process_first_frame, threshold_mask, clean_mask, get_net, clean_fullmask, get_minmax



def maskpc(mask,target_class_id):
    binary_mask = (mask == target_class_id)
    return binary_mask

def get_class(x_pixel, y_pixel, max_y, min_y, max_x, min_x): 
    if  min_y[0]-1 <= y_pixel < max_y[0]:  #base area2        
        class_id="base area2"
    elif min_y[1] <= y_pixel < max_y[1]:  #nomans land 2
        class_id="nomans land 2"
    elif min_y[2] <= y_pixel < max_y[2]:  #net area2
        class_id= "net area2"
    elif min_y[3] <= y_pixel < max_y[3]:  #net area1
        class_id= "net area1"
    elif min_y[4] <= y_pixel < max_y[4]: #nomansland1
        class_id="nomansland1"
    elif min_y[5] <= y_pixel < max_y[5]+1: #basearea1
        class_id="nomansland1"
    else:
        class_id="out of court"
    return class_id

#mpc=maskpc(full_cleaned_mask,i)



# def get_class(x_pixel, y_pixel, max_y, min_y, max_x, min_x):
#         regions = ["base area2", "nomans land 2", "net area2", "net area1", "nomansland1", "basearea1"]
#         for i in range(len(min_y)):
#             if min_y[i] <= y_pixel < max_y[i]:
#                 return regions[i]
#         return "out of court"

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

################################# Shot Detection
def shot_detection(device, video_path, shot_csv, max_y, min_y, max_x, min_x,):
    # try:
    #     torch.cuda.set_device(gpu_id)
    #     model = YOLO("/notebooks/shot_detection/runs/pose/train/weights/best.pt")
    #     model.to(f'cuda:{gpu_id}')
        
    try:
        model = YOLO("../shot_detection/runs/pose/train/weights/best.pt").to(device)
        # Process video or frames here
        results = model("video_path.mp4", stream=True)

        
        summary = {"frame": [], "confidence": [], "class": [], "X": [], "Y": [], "H": [], "W": [], "location": [],"X(m)": [], "Y(m)": []}
        results = model(video_path, stream=True)
        
        for idx, result in enumerate(results):
            boxes = result.boxes
            for i in range(boxes.conf.shape[0]):
                conf = boxes.conf[i].item()
                if conf < 0.35:
                    continue
                cls = int(boxes.cls[i].item())
                x, y, w, h = boxes.xywh[i].cpu().round().int().numpy()
                x_meter, y_meter = calculate_position(x, y, max_y, min_y, max_x, min_x)
                location=get_class(x, y, max_y, min_y, max_x, min_x)
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

# Ball Tracking
def ball_tracking(gpu_id,video_path,ball_csv, max_y, min_y, max_x, min_x,net_min,filename_without_extension,base_path):
    import numpy as np 
    try:
        torch.cuda.set_device(gpu_id)
        device = select_device(f'cuda:{gpu_id}')
        model = DetectMultiBackend('../train25/weights/best.pt', device=device)
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
                            csv_writer.writerow([dataset.frame, names[int(cls)], *xywh, x_meter, y_meter,location])
        
        print(f"Ball tracking completed, saved to {ball_csv}")
    
    
        df=pd.read_csv(ball_csv)
        #REMOVE STATIC BALLS
        print("removing static balls")

        # Filter out only 'ball' class data
        ball_data = df[df['class'] == 'ball']

        # Initialize a list to store indices of static balls to remove
        static_ball_indices = []
        dst=[]
        # Loop through frames, starting from the second frame
        for i in range(1, len(ball_data)):
            # Get the current and previous frame data
            prev_frame = ball_data.iloc[i - 1]['frame']
            curr_frame = ball_data.iloc[i]['frame']

            # Get balls for the current and previous frame
            prev_frame_balls = ball_data[ball_data['frame'] == prev_frame]
            curr_frame_balls = ball_data[ball_data['frame'] == curr_frame]

            # Check if there's only one ball in the current frame
            if len(curr_frame_balls) == 1:
                distance = np.sqrt((curr_frame_balls['X'].iloc[0] - prev_frame_balls['X'].iloc[0])**2 + 
                                    (curr_frame_balls['Y'].iloc[0] - prev_frame_balls['Y'].iloc[0])**2)
                if distance < 2:
                    static_ball_indices.append(curr_frame_balls.index[0])
            else:
                # Multiple balls in the current frame
                for _, curr_ball in curr_frame_balls.iterrows():
                    for _, prev_ball in prev_frame_balls.iterrows():
                        if curr_ball['frame'] != prev_ball['frame']:
                            distance = np.sqrt((curr_ball['X'] - prev_ball['X'])**2 + 
                                                (curr_ball['Y'] - prev_ball['Y'])**2)
                            # print(f'this is {distance}')
                            dst.append(f"for {curr_ball['frame']} and {prev_ball['frame']} , distance is {distance}")
                            if distance < 2:
                                # If any distance is less than 2, consider this ball as static
                                #print(curr_ball)
                                static_ball_indices.append(curr_ball.name)
                                break  # Move to the next ball in the current frame

        # Remove the static balls from the original dataset
        data_cleaned = df.drop(index=static_ball_indices)

        ######   ELIMINATE DUPLICATES
        print("eliminating duplicates")
        df_filtered=data_cleaned
        df_filtered['frame'] = df_filtered['frame'].astype(int)
        cc = []  # List to store the selected balls with max movement and non-duplicate balls
        processed_frames = []  # List to store the processed balls (both max and non-duplicate)

        # Iterate over frames (ensuring df_filtered is sorted by 'frame')
        for i in range(0, df_filtered['frame'].max()):
            #print(f'currently at index {i}')
            # Filter out out-of-bounds data
            df_filtered_valid = df_filtered[(df_filtered['X (m)'] != 'out of bounds') & 
                                            (df_filtered['Y (m)'] != 'out of bounds')]

            # Convert columns to numeric
            df_filtered_valid['Y (m)'] = df_filtered_valid['Y (m)'].astype(float)
            df_filtered_valid['X (m)'] = df_filtered_valid['X (m)'].astype(float)

            # Get balls for the current frame
            current_frame_balls = df_filtered_valid[df_filtered_valid["frame"] == i]

            # Check if the current frame contains any balls
            if current_frame_balls.empty:
                #print(f"Frame {i} is empty. Skipping...")
                continue  # Skip empty frames or handle them as needed

            # Check if there is more than one ball in the current frame
            if len(current_frame_balls) > 1:
                d = []  # List to hold the distances for each ball in the current frame

                # Loop through each ball in the current frame
                for j in range(len(current_frame_balls)):
                    current_ball = current_frame_balls.iloc[j]

                    # Attempt to find the ball in the previous frame
                    frame_found = False
                    for k in range(i-1, -1, -1):  # Loop backwards through frames
                        previous_frame = df_filtered_valid[(df_filtered_valid["frame"] == k) & (df_filtered_valid["class"] == "ball")]
                        if not previous_frame.empty:
                            previous_ball = previous_frame.iloc[0]  # Get the first detected ball from the previous frame
                            frame_found = True
                            break  # Stop once we find a valid previous frame

                    if frame_found:
                        # Calculate distance between the ball in the previous frame and the current ball
                        distance = np.sqrt((previous_ball['X (m)'] - current_ball['X (m)'])**2 + 
                                           (previous_ball['Y (m)'] - current_ball['Y (m)'])**2)
                        d.append(distance)
                    else:
                        d.append(np.nan)  # Append NaN if no previous frame with a ball was found

                # Find the index of the ball that moved the most (max distance)
                max_distance_index = np.argmax(d) if len(d) > 0 else None

                if max_distance_index is not None:
                    # Keep only the ball with the maximum movement (distance)
                    max_ball = current_frame_balls.iloc[max_distance_index]
                    cc.append(max_ball)
                    processed_frames.append(max_ball.to_dict())  # Convert max ball to dict and add


                #print(f"Processed frame {i}, max ball: {max_ball.name}")

            else:
                # If there is only one ball in the frame, add it to the results
                cc.append(current_frame_balls.iloc[0])
                processed_frames.append(current_frame_balls.iloc[0].to_dict())  # Convert to dict
                #print(f"Processed frame {i}, single ball: {current_frame_balls.iloc[0].name}")

        # Convert the list of processed balls into a DataFrame
        processed_df = pd.DataFrame(processed_frames)

        #########################SPEED 
        print("computing speed")

        df=processed_df
        import numpy as np
        # Filter the rows where Label == 'ball'
        # Filter the rows where Label == 'ball'
        ball_df = df[df['class'] == 'ball']

        # Initialize an empty list to store the speeds
        speeds_kmh = [None]  # The first entry will have no speed

        # Calculate the speed for each frame with a ball
        for i in range(1, len(ball_df)):
            x1, y1 = ball_df.iloc[i-1][['X (m)', 'Y (m)']]
            x2, y2 = ball_df.iloc[i][['X (m)', 'Y (m)']]
            frame1, frame2 = ball_df.iloc[i-1]['frame'], ball_df.iloc[i]['frame']

            # Calculate Euclidean distance in meters
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Calculate time difference in seconds
            time_diff = (frame2 - frame1) / 25  # 25 frames per second

            # Calculate speed in meters per second
            speed_mps = distance / time_diff

            # Convert speed to kilometers per hour (1 m/s = 3.6 km/h)
            speed_kmh = speed_mps * 3.6
            speeds_kmh.append(speed_kmh)

        # Now, assign the speeds to the original DataFrame where Label == 'ball'
        df['Speed (km/h)'] = np.nan  # Initialize the 'Speed (km/h)' column with NaNs
        df.loc[df['class'] == 'ball', 'Speed (km/h)'] = speeds_kmh  # Assign speeds only to rows with 'ball'

        ##################
        print("trajectoryyyyyyy")

        # Define a function to check the pixel distance and determine if a hit occurred
        def check_hits_and_classify(db):
            previous_x, previous_y = None, None

            # Initialize lists to store shot classification results, crossed net status, and ball hit status
            trajectory = []
            crossed_net_status = []
            ball_hit_status = []

            # Loop through the DataFrame
            for index, row in db.iterrows():
                # Shot classification logic (cross-court or down-the-line)
                if row['class'] == 'ball':  # Only consider rows where class is "ball"
                    
                    if previous_x is not None and previous_y is not None:
                        # Calculate the differences in coordinates (current - previous)
                        delta_x = row['X'] - previous_x
                        delta_y = row['Y'] - previous_y

                        # Compute the angle in degrees using atan2
                        angle_rad = np.arctan2(delta_y, delta_x)  # Get angle in radians
                        angle_deg = np.degrees(angle_rad)  # Convert to degrees

                        # Classify based on the angle
                        if (0 <= angle_deg <= 20) or (120 <= angle_deg <= 180):
                            trajectory.append("cross court")
                            
                        else:
                            trajectory.append("down the line")
                        
                    else:
                        # For the first detection, no previous coordinates, so no classification yet
                        trajectory.append(None)
                        
                    # Check if the ball has crossed the net (Y is between 113 and 204)
                    if net_min-30 <= row['Y'] <= net_min+50:
                        crossed_net_status.append(True)
                    else:
                        crossed_net_status.append(False)

#                     # For ball-hit detection, check if racket is in the same frame
#                     racket_row = db[(db['frame'] == row['frame']) & (db['class'] == 'racket')]
#                     if not racket_row.empty:
#                         racket_x_pixel, racket_y_pixel = racket_row['X'].values[0], racket_row['Y'].values[0]

#                         # Calculate the pixel distance between the ball and racket
#                         distance = np.sqrt((row['X'] - racket_x_pixel)**2 + (row['Y'] - racket_y_pixel)**2)

#                         # If the distance is less than 20 pixels, mark it as a hit
#                         if distance < 20:
#                             ball_hit_status.append("hit detected")
#                         else:
#                             ball_hit_status.append(None)
#                     else:
#                         ball_hit_status.append(None)

                    # Update the previous coordinates to the current ones
                    previous_x, previous_y = row['X'], row['Y']
                else:
                    # For rows where class is not "ball", append None for all categories
                    trajectory.append(None)
                    crossed_net_status.append(None)
                    #ball_hit_status.append(None)

            # Add the classification, crossed net status, and ball hit status to the DataFrame
            db['trajectory'] = trajectory
            db['crossed_net'] = crossed_net_status
            #db['ball_hit'] = ball_hit_status

            return db

        # Run the function to check for hits and classify shots
        db = check_hits_and_classify(df)

        ball_cleaned, _ = os.path.splitext(ball_csv)

        db.to_csv(os.path.join(base_path, f"{filename_without_extension}b.csv"), index=False)
        print("ball csv cleaned and saved ")
    except Exception as e:
        print(f"Ball tracking failed: {e}")

############################ Player ID Tracking
def player_id_tracking(device, video_path,player_csv, max_y, min_y, max_x, min_x,filename_without_extension,base_path):
    try:
        # torch.cuda.set_device(device)
        
        tracker = PersonTracker(
            reid_model_path='../LATransformer/model/la_with_lmbd_8/net_best.pth',
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
        df=pd.read_csv(player_csv)
        df=df[df["X(m)"]!= "out of court"]
        first_4_ids = df['ID'].drop_duplicates().head(4).tolist()
        #########################################ID FIXING

        # Dictionary to store the last known coordinates of each ID
        last_positions = {id: None for id in first_4_ids}

        for frame in sorted(df['Frame'].unique()):
            frame_df = df[df['Frame'] == frame]
            print(f"{frame} is {last_positions}")

            # Update last known positions for IDs in this frame
            for _, row in frame_df.iterrows():
                if row['ID'] in first_4_ids:
                    last_positions[row['ID']] = np.array([row['X'], row['Y']])

            # Check which IDs from first_4_ids are in this frame
            ids_in_frame = frame_df['ID'].isin(first_4_ids)

            if not ids_in_frame.all():  # If not all IDs in first_4_ids are in the frame
                assigned_ids = set()  # Keep track of already reassigned IDs in this frame

                for idx, (is_in, id) in enumerate(zip(ids_in_frame, frame_df['ID'])):
                    if not is_in:
                        # Find available IDs not in this frame or already reassigned
                        available_ids = [id for id in first_4_ids if id not in frame_df['ID'].values and id not in assigned_ids]

                        if available_ids:
                            # Compute distances using last known positions
                            distances = []
                            for available_id in available_ids:
                                if last_positions[available_id] is not None:  # Check if we have a last position
                                    current_coords = frame_df.iloc[idx][['X', 'Y']].values
                                    distance = np.linalg.norm(last_positions[available_id] - current_coords)
                                    distances.append((available_id, distance))

                            if distances:
                                # Assign the ID with the smallest distance
                                new_id, _ = min(distances, key=lambda x: x[1])
                                df.loc[frame_df.index[idx], 'ID'] = new_id
                                assigned_ids.add(new_id)  # Mark this ID as reassigned


        db = df[df['ID'].isin(first_4_ids)]
        csv_cleaned, _ = os.path.splitext(player_csv)
        os.path.join(base_path, f"{filename_without_extension}_s.csv")
        db.to_csv(os.path.join(base_path, f"{filename_without_extension}p.csv"), index=False)
        
        print(f"Player ID tracking completed, saved to {player_csv}")
    except Exception as e:
        print(f"Player ID tracking failed: {e}")

class PersonDetector:
    def __init__(self, model_size, device='cuda:0' if torch.cuda.device_count() > 0 else 'cpu'):
        # self.device='cuda:0' if num_gpus > 0 else 'cpu'
        self.model = YOLO(model_size).to(device)
    
    def detect(self, frame):
        results = self.model(frame, classes=0)  # Detect persons only
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if box.conf[0] > 0.5:  # Confidence threshold
                    # Convert map object to tuple
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
        self.features = [feature]  # Simplified for this example
        self.missed_frames = 0
    def update(self, bbox, feature):
        self.bbox = bbox
        self.features.append(feature)
        self.missed_frames = 0
    def mark_missed(self):
        self.missed_frames += 1

# class PersonTracker:
#     def __init__(self, reid_model_path, yolo_model="yolov8n.pt", device='cpu'):
#         self.device = device
#         self.detector = PersonDetector(yolo_model, device)
#         self.tracks = {}
#         self.next_id = 1        
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
        features = [self.reid_model.extract_features(frame[y1:y2, x1:x2]) for x1, y1, x2, y2 in detections if frame[y1:y2, x1:x2].size > 0]
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

# Main execution
#if __name__ == "__main__":
    
def process_vid(video_path):
    #video_path = '/notebooks/test_cases/20241204T220001.mkv'  #'/notebooks/20240725T203000_Trimm.mp4'#
    
    #video_path="/notebooks/20240815T194501.mkv"
    # Paths and constants
    unique_id = str(time.time()).replace('.', '')
    base_path = f"results__{unique_id}"
    filename_with_extension = os.path.basename(video_path)  # Extracts the file name with extension
    filename_without_extension = os.path.splitext(filename_with_extension)[0]  # Removes the extension
    base_path = f"results__{filename_without_extension}_{unique_id}"
    ball_csv = os.path.join(base_path, f"{filename_without_extension}_ball.csv")
    shot_csv=os.path.join(base_path, f"{filename_without_extension}_s.csv")
    player_csv = os.path.join(base_path, f"{filename_without_extension}_player.csv")
    os.makedirs(base_path, exist_ok=True)


        # Shared segmentation data (computed once)
    video = cv2.VideoCapture(video_path)
    pp = process_first_frame(video_path)
    cleaned_mask = threshold_mask(pp, target_class_id=3, threshold=0.5)
    cleaned_mask = clean_mask(cleaned_mask)
    net_p, net_min = get_net(cleaned_mask)
    full_cleaned_mask = clean_fullmask(pp, num_classes=7, threshold=0.5, kernel_size=3)
    def maskpc(mask,target_class_id):
        binary_mask = (mask == target_class_id)
        return binary_mask
    full=[2,7,5,4,6,1]
    cls=[7,5,4,6,1]
    i=2
    max_y = []
    min_y = []
    max_x = []
    min_x = []

    mpc=maskpc(full_cleaned_mask,i)
    max_y_j, min_y_j, max_x_j, min_x_j = get_minmax(mpc)
    max_y.append(max_y_j)
    min_y.append(min_y_j)
    max_x.append(max_x_j)
    min_x.append(min_x_j)
    for i in range(len(cls)):
        j=cls[i]
        mpc=maskpc(full_cleaned_mask,j)
        max_y_j, min_y_j, max_x_j, min_x_j = get_minmax(mpc)  # Call the function for pp and store results
        #print(max_y_j)

        min_y_j=max_y[-1]
        max_y.append(max_y_j)
        min_y.append(min_y_j)
        max_x.append(max_x_j)
        min_x.append(min_x_j)

    print(f" max x is   {max_x}")
    print(f" min x is   {min_x}")



    # Your existing initialization code
    stime = time.time()
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    device = 'cuda:0' if num_gpus > 0 else 'cpu'
     # Create a multiprocessing context with 'spawn'
    ctx = get_context('spawn')
    # Pass device string to each process
    # processes = [
    #     Process(target=shot_detection, args=(device, video_path,shot_csv, max_y, min_y, max_x, min_x)),
    #     Process(target=ball_tracking, args=(device, video_path,ball_csv, max_y, min_y, max_x, min_x,net_min)),
    #     Process(target=player_id_tracking, args=(device, video_path,player_csv, max_y, min_y, max_x, min_x))
    # ]
    processes = [
        ctx.Process(target=shot_detection, args=(device, video_path, shot_csv, max_y, min_y, max_x, min_x)),
        ctx.Process(target=ball_tracking, args=(device, video_path, ball_csv, max_y, min_y, max_x, min_x, net_min,filename_without_extension,base_path)),
        ctx.Process(target=player_id_tracking, args=(device, video_path, player_csv, max_y, min_y, max_x, min_x,filename_without_extension,base_path))
    ]

     
  
    # Start all processes
    for p in processes:
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    etime = time.time()
    print(f"Elapsed time: {etime - stime} seconds")
    player_cleaned_csv = os.path.join(base_path, f"{filename_without_extension}p.csv")
    #f"{csv_cleaned}_cleaned.csv"
    ball_cleaned_csv=os.path.join(base_path, f"{filename_without_extension}b.csv")
    #db.to_csv(f"{ball_cleaned}_cleaned.csv", index=False)
    # Return the paths to the CSV files
    print(f"done processing and saved to {base_path}")

    return ball_cleaned_csv, shot_csv, player_cleaned_csv

    

 #Gradio interface
iface = gr.Interface(
    fn=process_vid,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.File(label="Ball Detections"),
        gr.File(label="Shot Detections"),
        gr.File(label="Player Tracking")
    ],
    title="Video Processing App",
    description="Upload a video to process for ball detections, shot detections, and player tracking. Results will be available for download as CSV files.",
    #queue=True  # Enable queuing
)


    

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        print("Start method already set, proceeding with:", multiprocessing.get_start_method())
    #process_vid("/notebooks/20240725T203000_Trimm.mp4")
    iface.launch(share=True)