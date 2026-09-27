# NOTE BEFORE RUNNING:
# If you haven't already, you must install CLIP manually in the Pi terminal to prevent Ultralytics from freezing:
# sudo apt-get install git
# pip install git+https://github.com/ultralytics/CLIP.git

import time
import os
import cv2
import numpy as np
import threading
import requests
import mediapipe as mp
from collections import deque
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import sys

def get_path(filename):
    """Dynamically routes paths for both normal Python execution and PyInstaller bundles."""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller extracts data files to this temporary _MEIPASS folder
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

# ==========================================
# 0. CONFIGURATION & NETWORK SYNC
# ==========================================
# Replace with your computer's local IP address where server.py is running
SERVER_URL = "http://10.43.147.168:3000/"

class WebcamStream:
    # OPTIMIZATION 1: Lowered capture resolution to 720p to save Pi memory/CPU
    def __init__(self, src=0, width=1920, height=1080):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 60) 
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# 1. SETUP & YOLO-WORLD MODEL LOADING
# ==========================================
fps_times = deque(maxlen=30)
current_fps = 0.0

yolo_path = get_path("yolov8s-world.pt")
print("Loading YOLO-World open-vocabulary model on Raspberry Pi...")
yolo_model = YOLO(yolo_path)
print("YOLO-World loaded successfully.")

active_text_prompts = ["lamp", "bottle", "cup"]
yolo_model.set_classes(active_text_prompts)

TRACKER_IDS = [0, 1]
prev_wrist_coords = {t: None for t in TRACKER_IDS}

aim_start_time = {t: None for t in TRACKER_IDS}
cooldown_end_time = {t: 0.0 for t in TRACKER_IDS}
dwell_anchor = {t: None for t in TRACKER_IDS}
DWELL_PIXEL_RADIUS = 40  

system_timer = 0
status_text = "Waiting for hands..."
status_color = (0, 0, 255)

calibrated_objects = {}  
active_yolo_boxes = []   
current_ray_end = None   
waiting_for_point = False
pending_calibration_name = None
last_printed_target = None

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# --- Async MediaPipe Setup ---
latest_mp_result = None
def mp_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_mp_result
    latest_mp_result = result

model_path = get_path("hand_landmarker.task")
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM, 
    num_hands=2,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2,
    result_callback=mp_callback
)
detector = vision.HandLandmarker.create_from_options(options)

def calc_dist(lm1, lm2):
    return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)

def get_calibrated_class_names():
    names = []
    for obj in calibrated_objects.values():
        if obj["class_name"] not in names:
            names.append(obj["class_name"])
    return names

last_scan_classes = None

# --- Async YOLO-World Setup ---

yolo_input_frame = None
yolo_desired_classes = None
yolo_new_frame_event = threading.Event()
latest_yolo_boxes = []
yolo_worker_running = True

def yolo_worker_loop():
    global latest_yolo_boxes, last_scan_classes
    while yolo_worker_running:
        if not yolo_new_frame_event.wait(timeout=0.2):
            continue
        yolo_new_frame_event.clear()

        frame_for_yolo = yolo_input_frame
        desired_classes = yolo_desired_classes
        if frame_for_yolo is None:
            continue

        if desired_classes != last_scan_classes:
            if desired_classes:
                yolo_model.set_classes(desired_classes)
            last_scan_classes = list(desired_classes) if desired_classes else None

        if not desired_classes:
            latest_yolo_boxes = []
            continue

        yolo_results = yolo_model(frame_for_yolo, verbose=False)[0]
        boxes = []
        if yolo_results.boxes is not None:
            for box in yolo_results.boxes:
                cls_id = int(box.cls[0])
                class_name = yolo_model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                boxes.append({
                    "box": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "conf": conf
                })
        latest_yolo_boxes = boxes

threading.Thread(target=yolo_worker_loop, daemon=True).start()

# --- Background Server Sync Thread ---
def sync_with_server_loop():
    global waiting_for_point, pending_calibration_name, calibrated_objects, status_text
    while yolo_worker_running:
        try:
            payload = {
                "calibrated_objects": {k: v for k, v in calibrated_objects.items()},
                "status_text": status_text,
                "active_prompts": active_text_prompts
            }
            res = requests.post(f"{SERVER_URL}/sync_state", json=payload, timeout=1.0)
            if res.status_code == 200:
                data = res.json()
                if data.get("clear_requested"):
                    calibrated_objects.clear()
                    print("[Pi Client] Cleared all calibrated objects via server command.")
                
                # Check if web dashboard triggered calibration mode
                server_waiting = data.get("waiting_for_point", False)
                if server_waiting and not waiting_for_point:
                    waiting_for_point = True
                    pending_calibration_name = data.get("pending_calibration_name")
                    print(f"[Pi Client] Calibration armed for '{pending_calibration_name}' from web UI.")
                elif not server_waiting and waiting_for_point:
                    waiting_for_point = False
                    pending_calibration_name = None
        except Exception:
            # Silent catch for network hiccups between Pi and PC
            pass
        time.sleep(0.5)

threading.Thread(target=sync_with_server_loop, daemon=True).start()

# OPTIMIZATION 2: Async Frame Streamer Thread
stream_frame = None
stream_event = threading.Event()

def frame_streaming_loop():
    global stream_frame
    while yolo_worker_running:
        if stream_event.wait(timeout=0.1):
            stream_event.clear()
            frame_to_send = stream_frame
            if frame_to_send is not None:
                try:
                    # Downscale the frame slightly for the web UI to save network bandwidth
                    small_frame = cv2.resize(frame_to_send, (640, 360))
                    success, encoded_image = cv2.imencode(".jpg", small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    if success:
                        requests.post(f"{SERVER_URL}/update_frame", files={"frame": encoded_image.tobytes()}, timeout=0.2)
                except Exception:
                    pass

threading.Thread(target=frame_streaming_loop, daemon=True).start()

def is_line_intersecting_box(p1, p2, box):
    x1, y1, x2, y2 = box
    if (x1 <= p1[0] <= x2 and y1 <= p1[1] <= y2) or (x1 <= p2[0] <= x2 and y1 <= p2[1] <= y2):
        return True
    for t in np.linspace(0, 1, 20):
        x = int(p1[0] + t * (p2[0] - p1[0]))
        y = int(p1[1] + t * (p2[1] - p1[1]))
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

# ==========================================
# 2. MAIN LOOP
# ==========================================
cap = WebcamStream(src=0).start()
print("ShellHacks 2026 - Raspberry Pi Client Initialized.")
print("Running in headless mode. Press Ctrl+C in the terminal to exit.")

last_timestamp_ms = 0  # Initializes the timestamp tracker to prevent MediaPipe crashes

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame.copy()
        h, w, _ = frame.shape
        
        current_time = time.time()
        fps_times.append(current_time)
        if len(fps_times) > 1:
            current_fps = len(fps_times) / (fps_times[-1] - fps_times[0])

        if waiting_for_point:
            desired_classes = active_text_prompts
        else:
            desired_classes = get_calibrated_class_names()

        yolo_input_frame = frame.copy()
        yolo_desired_classes = desired_classes
        yolo_new_frame_event.set()

        active_yolo_boxes = latest_yolo_boxes

        # ==========================================
        # MEDIAPIPE ASYNC DETECTION WITH TIMESTAMP ENFORCEMENT
        # ==========================================
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        current_time_ms = int(current_time * 1000)
        
        # Force the timestamp to be strictly increasing
        if current_time_ms <= last_timestamp_ms:
            current_time_ms = last_timestamp_ms + 1
        last_timestamp_ms = current_time_ms
        
        detector.detect_async(mp_image, current_time_ms)

        if system_timer > 0:
            system_timer -= 1

        current_ray_end = None
        aimed_object_this_frame = None
        overlapped_yolo_indices = set()
        overlapped_calibrated_names = set()

        # ==========================================
        # 3. SPATIAL TRACKING & RAY INTERCEPTION
        # ==========================================
        current_frame_hands = []
        current_result = latest_mp_result 
        
        if current_result and current_result.hand_landmarks:
            for idx, hand in enumerate(current_result.hand_landmarks):
                wrist = hand[0]
                pixel_points = []
                
                for lm in hand:
                    px, py = int(lm.x * w), int(lm.y * h)
                    pixel_points.append((px, py))
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)
                    
                for p1, p2 in HAND_CONNECTIONS:
                    if p1 < len(pixel_points) and p2 < len(pixel_points):
                        cv2.line(frame, pixel_points[p1], pixel_points[p2], (255, 0, 0), 2)
                
                index_extended = calc_dist(hand[8], wrist) > calc_dist(hand[6], wrist)
                middle_curled = calc_dist(hand[12], wrist) < calc_dist(hand[10], wrist)
                ring_curled = calc_dist(hand[16], wrist) < calc_dist(hand[14], wrist)
                pinky_curled = calc_dist(hand[20], wrist) < calc_dist(hand[18], wrist)
                
                is_pointing = index_extended and middle_curled and ring_curled and pinky_curled

                current_frame_hands.append({
                    "wrist": np.array([wrist.x, wrist.y, wrist.z]),
                    "index_tip_px": np.array([pixel_points[8][0], pixel_points[8][1]]),
                    "index_mcp_px": np.array([pixel_points[5][0], pixel_points[5][1]]),
                    "is_pointing": is_pointing
                })

        assigned_mapping = {}
        unassigned_indices = list(range(len(current_frame_hands)))

        for t in TRACKER_IDS:
            if prev_wrist_coords[t] is not None:
                best_idx, min_dist = None, float('inf')
                for idx in unassigned_indices:
                    dist = np.linalg.norm(current_frame_hands[idx]["wrist"] - prev_wrist_coords[t])
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                if best_idx is not None and min_dist < 0.3:
                    assigned_mapping[t] = current_frame_hands[best_idx]
                    unassigned_indices.remove(best_idx)

        for idx in unassigned_indices:
            for t in TRACKER_IDS:
                if t not in assigned_mapping:
                    assigned_mapping[t] = current_frame_hands[idx]
                    break

        hands_present = []
        for t, hand_data in assigned_mapping.items():
            hands_present.append(t)
            prev_wrist_coords[t] = hand_data["wrist"]

        for t in TRACKER_IDS:
            if t not in hands_present:
                prev_wrist_coords[t] = None
                aim_start_time[t] = None
                cooldown_end_time[t] = 0.0
                dwell_anchor[t] = None

        for t, hand_data in assigned_mapping.items():
            tip_px = hand_data["index_tip_px"]
            mcp_px = hand_data["index_mcp_px"]
            is_pointing = hand_data["is_pointing"]
            
            if is_pointing:
                dx = tip_px[0] - mcp_px[0]
                dy = tip_px[1] - mcp_px[1]
                ray_length = 50
                end_x = int(tip_px[0] + dx * ray_length)
                end_y = int(tip_px[1] + dy * ray_length)
                
                ray_p1 = tuple(tip_px)
                ray_p2 = (end_x, end_y)
                current_ray_end = ray_p2
                
                cv2.line(frame, ray_p1, ray_p2, (200, 200, 200), 2)
                
                if waiting_for_point:
                    for idx, obj in enumerate(active_yolo_boxes):
                        if is_line_intersecting_box(ray_p1, ray_p2, obj["box"]):
                            calibrated_objects[pending_calibration_name] = {
                                "box": list(obj["box"]),
                                "class_name": obj["class_name"]
                            }
                            if pending_calibration_name not in active_text_prompts:
                                active_text_prompts.append(pending_calibration_name)
                            
                            system_timer = 30
                            status_text = f"LOCKED OVERLAP: {pending_calibration_name.upper()}"
                            status_color = (255, 255, 0)
                            print(f"Calibration successful! Locked '{pending_calibration_name}' to bounds {obj['box']}")
                            
                            try:
                                requests.post(f"{SERVER_URL}/cancel", timeout=0.5)
                            except Exception:
                                pass
                                
                            waiting_for_point = False
                            pending_calibration_name = None
                            break

                for idx, obj in enumerate(active_yolo_boxes):
                    if is_line_intersecting_box(ray_p1, ray_p2, obj["box"]):
                        overlapped_yolo_indices.add(idx)

                for custom_name, data in calibrated_objects.items():
                    if is_line_intersecting_box(ray_p1, ray_p2, data["box"]):
                        overlapped_calibrated_names.add(custom_name)
                        aimed_object_this_frame = custom_name

            if not is_pointing:
                aim_start_time[t] = None
                cooldown_end_time[t] = 0.0
                dwell_anchor[t] = None
                continue

            if current_time < cooldown_end_time[t]:
                remaining = cooldown_end_time[t] - current_time
                progress = remaining / 7.0
                radius = int(progress * 25)
                if radius > 0:
                    cv2.circle(frame, tuple(tip_px), radius, (0, 0, 255), 2) 
                dwell_anchor[t] = None 
                continue
                
            if dwell_anchor[t] is None:
                dwell_anchor[t] = tip_px
                aim_start_time[t] = current_time
            else:
                drift = np.linalg.norm(tip_px - dwell_anchor[t])
                if drift > DWELL_PIXEL_RADIUS:
                    dwell_anchor[t] = tip_px         
                    aim_start_time[t] = current_time 

            elapsed_time = current_time - aim_start_time[t]
            progress = min(1.0, elapsed_time / 0.2)
            radius = int(progress * 25)
            
            if radius > 0:
                cv2.circle(frame, tuple(tip_px), radius, (255, 0, 255), 3) 

            if elapsed_time >= 0.2:
                cooldown_end_time[t] = current_time + 7.0 
                aim_start_time[t] = None
                dwell_anchor[t] = None
                
                cv2.circle(frame, tuple(tip_px), 30, (0, 255, 0), -1)
                if current_ray_end:
                    cv2.line(frame, tuple(tip_px), current_ray_end, (0, 255, 0), 6)
                
                system_timer = 15
                status_text = f"CLICKED!"
                status_color = (0, 255, 0)
                
                if aimed_object_this_frame:
                    print(f"\n[Pi Client] Target Hit: Pinging Server to turn on {aimed_object_this_frame}")
                    
                    def send_trigger():
                        try:
                            requests.post(f"{SERVER_URL}/trigger_command", json={
                                "device": aimed_object_this_frame,
                                "action": "turn on"
                            }, timeout=2.0)
                        except requests.exceptions.RequestException as e:
                            print(f"[Pi Client] Network error reaching server: {e}")
                            
                    threading.Thread(target=send_trigger, daemon=True).start()

        # ==========================================
        # 4. RENDER OVERLAYS
        # ==========================================
        for idx, obj in enumerate(active_yolo_boxes):
            x1, y1, x2, y2 = obj["box"]
            box_color = (0, 140, 255) if idx in overlapped_yolo_indices else (255, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{obj['class_name']} ({int(obj['conf']*100)}%)", (x1, y1 - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        for custom_name, data in calibrated_objects.items():
            x1, y1, x2, y2 = data["box"]
            cal_box_color = (255, 0, 255) if custom_name in overlapped_calibrated_names else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), cal_box_color, 3)
            cv2.putText(frame, f"[{custom_name}]", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cal_box_color, 2)

        if waiting_for_point:
            status_text = f"ARMED: Point ray at '{pending_calibration_name}'"
            status_color = (0, 255, 255)

        cv2.putText(frame, f"FPS: {current_fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # OPTIMIZATION 3: Async Frame Transfer & Headless Execution
        # Hand off the fully drawn frame to the streaming thread without pausing the camera loop
        stream_frame = frame.copy()
        stream_event.set()

        # Local window rendering disabled to save Pi GUI overhead. 
        # Stream is securely routed to web app.

except KeyboardInterrupt:
    print("\n[Pi Client] Shutting down gracefully...")

finally:
    yolo_worker_running = False
    cap.stop()
    cv2.destroyAllWindows()