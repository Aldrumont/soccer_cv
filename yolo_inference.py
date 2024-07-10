import cv2
import numpy as np
import norfair
import torch
from norfair import AbsolutePaths, Paths, Video
from norfair.camera_motion import HomographyTransformationGetter, MotionEstimator
from norfair.distances import create_normalized_mean_euclidean_distance
import random


tracker_person = norfair.Tracker(distance_function="euclidean", distance_threshold=100, past_detections_length=30)
tracker_ball = norfair.Tracker(distance_function="euclidean", distance_threshold=100, past_detections_length=30, hit_counter_max=500)

transformations_getter = HomographyTransformationGetter()
# paths_drawer = AbsolutePaths(max_history=50, thickness=2)
motion_estimator = MotionEstimator(
    max_points=500, min_distance=7, transformations_getter=transformations_getter
)
yolo_version = 8
last_team_with_ball = None

ignore_predict_area = {'x1':0, 'y1':0, 'x2': 1900, 'y2': 200}

if yolo_version == 5:
    # Load YOLOv5 model
    model_person = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model_ball = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5-ball.pt")
elif yolo_version == 8:
    from ultralytics import YOLO
    # model_ball = YOLO('ball.pt')
    # model_person = YOLO('yolov8s.pt')
    model_person = YOLO('yolov8m-football.pt')
    model_ball = YOLO('yolov8m-football.pt')
    # model_pose = YOLO('yolov8n-pose.pt')


# Video path
video_path = 'final_champions.mp4'

class_main_hsv_color = {
    "Borussia": (51,87,96),
    "Real Madrid": (220, 0, 100),
    "Referee": (195,70,88),
    "Borrusia_GK": (12, 80, 98),
    "Real Madrid_GK": (78, 80, 87),
}

class_hsv_range = {
    "Borussia": [[31,48], [50,100], [70,100]],
    "Real Madrid": [[0,360], [0,15], [85,100]],
    "Borrusia_GK": [[0,10], [70,94], [84,100]],
    "Referee": [[119,149], [50,100], [50,100]],
    "Real Madrid_GK": [[54,66], [61,100], [61,100]],
}

ignore_pct_dict = {"top": 0.1, "bottom": 0.5, "left": 0.0, "right": 0.0}

def hsv2rgb(hsv):
    h, s, v = hsv
    h = float(h)
    s = float(s) / 100
    v = float(v) / 100

    if s == 0:
        r = g = b = int(v * 255)
    else:
        h = h / 60
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q

        r, g, b = int(r * 255), int(g * 255), int(b * 255)

    return (r, g, b)

def show_pixels_in_range(image, hsv_range, binary=True):
    lower_h = int((hsv_range[0][0] / 360) * 255)
    upper_h = int((hsv_range[0][1] / 360) * 255)
    lower_s = int((hsv_range[1][0] / 100) * 255)
    upper_s = int((hsv_range[1][1] / 100) * 255)
    lower_v = int((hsv_range[2][0] / 100) * 255)
    upper_v = int((hsv_range[2][1] / 100) * 255)

    lower = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
    upper = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower, upper)

    if binary:
        return mask

    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def class_criteria(label_dict):
    priority = ["Referee", "Borrusia_GK", "Real Madrid_GK"]
    aux_dict = label_dict.copy()
    for p in priority:
        label_dict[p] *=10

    return max(label_dict, key=label_dict.get)

def determine_class(image, class_hsv_range, ignore_pct_dict):
    height, width, _ = image.shape
    x1, x2, y1, y2 = int(ignore_pct_dict["left"] * width), int((1 - ignore_pct_dict["right"]) * width), int(ignore_pct_dict["top"] * height), int((1 - ignore_pct_dict["bottom"]) * height)
    cropped_image = image[y1:y2, x1:x2]
    label_dict = {}
    for class_name in class_hsv_range:
        img = show_pixels_in_range(cropped_image, class_hsv_range[class_name])
        mean = np.mean(img) / 255
        label_dict[class_name] = mean
    
    # print(label_dict)
    label = class_criteria(label_dict)
    return label

      

def process_frame(frame, track_history_person={}, track_history_ball={}):
    global last_team_with_ball
    
    results_person = model_person(frame)
    results_ball = model_ball(frame)
    # results_pose = model_pose(frame)
    detections_person = []
    detections_ball = []
    ball_bbox = None

    # Process person detections
    yolo_results = []
    if yolo_version == 5:
        for result in results_person.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result.int().cpu().tolist()
            yolo_results.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "cls": 98})
        for result in results_ball.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result.int().cpu().tolist()
            yolo_results.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "cls": 99})

    elif yolo_version == 8:
        for result in results_person[0].boxes:
            x1, y1, x2, y2, conf, cls = result.data.int().cpu().tolist()[0]
            if cls != 0:
                yolo_results.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "cls": 98})
        for result in results_ball[0].boxes:
            x1, y1, x2, y2, conf, cls = result.data.int().cpu().tolist()[0]
            if cls == 0:
                yolo_results.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "cls": 99})
            
                
    for result in yolo_results:
        x1, y1, x2, y2, conf, cls = result["x1"], result["y1"], result["x2"], result["y2"], result["conf"], result["cls"]
        if cls == 98:  # Assuming person class is 0
            bbox = np.array([x1, y1, x2, y2])
            detections_person.append(norfair.Detection(points=np.array([[x1, y1], [x2, y2]]), data=bbox))
        if cls == 99: # Assuming ball class is 32
            ball_centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            # if y of ball_centroid is < ignore_predict_area['y2'] then ignore the ball
            if ball_centroid[1] < ignore_predict_area['y2']:
                continue
            ball_bbox = np.array([x1, y1, x2, y2])
            # draw a rectangle around the ball
            detections_ball.append(norfair.Detection(points=np.array([[x1, y1], [x2, y2]]), data=ball_bbox))
    
        
    mask = np.ones(frame.shape[:2], frame.dtype)
    for det in detections_person:
        i = det.points.astype(int)
        mask[i[0, 1] : i[1, 1], i[0, 0] : i[1, 0]] = 0
    coord_transformations = motion_estimator.update(frame, mask)
    tracked_objects_person = tracker_person.update(detections=detections_person, coord_transformations=coord_transformations)
    tracked_objects_ball = tracker_ball.update(detections=detections_ball, coord_transformations=coord_transformations)
    # frame = paths_drawer.draw(frame, tracked_objects_person, coord_transformations)
    
    player_with_ball = {"track_id": None, "distance_to_ball": 999999999}
    
    for tracked_object_person in tracked_objects_person:
        track_id = tracked_object_person.id
        if not tracked_object_person.live_points.any():
            continue
        if track_id not in track_history_person:
            track_history_person[track_id] = {"centroid": [], "color": []}
        points = tracked_object_person.get_estimate(absolute=True)
        
        x1, y1, x2, y2 = coord_transformations.abs_to_rel(points).astype(np.int32).flatten()
        person_image = frame[y1:y2, x1:x2]
        height, width, _ = person_image.shape
        if height <= 10 or width <= 10:
            continue
        
        label = determine_class(person_image, class_hsv_range, ignore_pct_dict)
        color = hsv2rgb(class_main_hsv_color[label])
        color = tuple(map(int, color[::-1]))
        track_history_person[track_id]["color"].append(color)
        
        feet_area = np.array([x1, y2*0.98, x2, y2], dtype=np.int32)
        #calculate the distance between the person and the ball
        if ball_bbox is not None:
            centroid_feet_area = np.array([np.mean(feet_area[[0, 2]]), np.mean(feet_area[[1, 3]])])
            ball_centroid = np.array([(ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2])
            distance_to_ball = np.linalg.norm(centroid_feet_area - ball_centroid)
            # draw a line between the person and the ball
            # cv2.line(frame, tuple(centroid_feet_area.astype(int)), tuple(ball_centroid.astype(int)), color, 2)
            if distance_to_ball < player_with_ball["distance_to_ball"]:
                player_with_ball["track_id"] = track_id
                player_with_ball["distance_to_ball"] = distance_to_ball
            
                # draw a triangle on person bbox like a indicator
                triangle_center_coord = ((x1 + x2) // 2, y1-3)
                triangle_left_coord = (triangle_center_coord[0] - 10, triangle_center_coord[1] - 15)
                triangle_right_coord = (triangle_center_coord[0] + 10, triangle_center_coord[1] - 15)
                triangle_vertices = np.array([triangle_center_coord, triangle_left_coord, triangle_right_coord], dtype=np.int32)
                player_with_ball['triangle_vertices'] = triangle_vertices
                player_with_ball['color'] = color
                player_with_ball['label'] = label
                # cv2.polylines(frame, [triangle_vertices], isClosed=True, color=(0, 255, 0), thickness=2)
        
        
        centroid = [np.mean(tracked_object_person.get_estimate(absolute=True), axis=0)]
        track_history_person[track_id]["centroid"].append(centroid)
        track_history_person[track_id]["centroid"] = track_history_person[track_id]["centroid"][-30:]
        # Color will be the majority color in the last 15 frames
        # color = max(set(track_history_person[track_id]["color"]), key=track_history_person[track_id]["color"].count)
        
        # for point in coord_transformations.abs_to_rel(centroid):
        #     cv2.circle(frame, tuple(point.astype(int)), 5, color, -1)
        
        # Draw the tracking lines
        points = np.vstack(track_history_person[track_id]['centroid']).astype(np.float32)
        fix_points = coord_transformations.abs_to_rel(points).astype(np.int32)
        if len(fix_points) > 1:
            cv2.polylines(frame, [fix_points], isClosed=False, color=color, thickness=1)
        
        # Draw rectangle around the person
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
        # # Draw the track id
        # cv2.putText(frame, str(track_id), tuple(fix_points[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    for tracked_object_ball in tracked_objects_ball:
        track_id = tracked_object_ball.id
        if not tracked_object_ball.live_points.any():
            continue
        if track_id not in track_history_ball:
            track_history_ball[track_id] = {"centroid": [], "color": []}
        points = tracked_object_ball.get_estimate(absolute=True)
        x1, y1, x2, y2 = coord_transformations.abs_to_rel(points).astype(np.int32).flatten()
        
        # Draw rectangle around the ball
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        

        
        ball_image = frame[y1:y2, x1:x2]
        height, width, _ = ball_image.shape
        if height <= 10 or width <= 10:
            continue
        
        color = (0, 255, 0)
        track_history_ball[track_id]["color"].append(color)
        
        centroid = [np.mean(tracked_object_ball.get_estimate(absolute=True), axis=0)]
        track_history_ball[track_id]["centroid"].append(centroid)
        track_history_ball[track_id]["centroid"] = track_history_ball[track_id]["centroid"][-30:]
        # Color will be the majority color in the last 15 frames
        color = max(set(track_history_ball[track_id]["color"]), key=track_history_ball[track_id]["color"].count)
        
        for point in coord_transformations.abs_to_rel(centroid):
            cv2.circle(frame, tuple(point.astype(int)), 5, color, -1)
        
        # Draw the tracking lines
        points = np.vstack(track_history_ball[track_id]['centroid']).astype(np.float32)
        fix_points = coord_transformations.abs_to_rel(points).astype(np.int32)
        if len(fix_points) > 1:
            cv2.polylines(frame, [fix_points], isClosed=False, color=color, thickness=3)
            
        # # Draw the track id
        # cv2.putText(frame, str(track_id), tuple(fix_points[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if player_with_ball['track_id'] is not None:
        cv2.fillPoly(frame, [player_with_ball['triangle_vertices']], player_with_ball['color'])
        last_team_with_ball = player_with_ball['label'] if player_with_ball['label'] != 'Referee' else last_team_with_ball
    
    cv2.putText(frame, f"Team possessing the ball: {last_team_with_ball}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame
        
            
            
        
        


def read_frames_from_video(video_path, output_path, show_video=False, save_video=False, skip_frames=1, start_time=0, end_time=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    adjusted_fps = fps / skip_frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if save_video:
        out = cv2.VideoWriter(output_path, fourcc, adjusted_fps, (width, height))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Convert start and end times to frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else total_frames

    if start_time > duration or (end_time and end_time > duration):
        print("Error: Specified time range exceeds video duration.")
        cap.release()
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    track_history_person = {}
    track_history_ball = {}
    
    frame_count = start_frame
    while frame_count < end_frame:
        ret, frame = cap.read()
        frame_count += 1
        print(f"Processing frame {frame_count}/{end_frame}")
        if frame_count % skip_frames != 0:
            continue
            
        if not ret:
            break
        processed_frame = process_frame(frame, track_history_person, track_history_ball)
        if processed_frame is not None:
            
            if show_video:
                cv2.imshow('Processed Frame', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            if save_video:
                out.write(processed_frame)
        
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()
# Example usage
output_path = 'processed_video.mp4'
read_frames_from_video(video_path, output_path, show_video=True, save_video=True, skip_frames=1, start_time=60*14, end_time=60*16)
output_path = 'processed_video2.mp4'
read_frames_from_video(video_path, output_path, show_video=True, save_video=True, skip_frames=1, start_time=60*25, end_time=60*28)
output_path = 'processed_video3.mp4'
read_frames_from_video(video_path, output_path, show_video=True, save_video=True, skip_frames=1, start_time=60*84, end_time=60*88)
