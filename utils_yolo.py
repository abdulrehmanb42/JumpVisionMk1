import cv2
import numpy as np
from rtmlib import Body
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Try to import YOLO, fallback to standard model if custom model fails
try:
    from ultralytics import YOLO
    # Use standard YOLO model for person detection with GPU
    yolo_model = YOLO('yolov8n.pt')  # Use nano model for speed
    # Move to GPU if available
    if hasattr(yolo_model, 'to'):
        yolo_model.to('cuda')
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    yolo_model = None

# Pose estimation setup - Use GPU
openpose_skeleton = True  # True for openpose-style, False for mmpose-style
device = 'cuda'  # Changed from 'mps' to 'cuda' for GPU acceleration
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
ankle_indices = [10, 13]
body = Body(to_openpose=openpose_skeleton, mode='balanced', backend=backend, device=device)

g = 9.810665  # m/s²


def free_fall(t, h0):
    return 0.5 * g * t ** 2 + h0


def cal_height(t):
    return (0.5 * g * (t / 2) ** 2) * 100


def detect_person(frame):
    """Detect person using YOLO and return bounding box"""
    if yolo_model is None:
        # Fallback: use the entire frame if YOLO is not available
        return (0, 0, frame.shape[1], frame.shape[0]), 1.0
    
    results = yolo_model(frame, verbose=False, device=0)  # Use GPU device 0
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Check if it's a person (class 0 in COCO dataset)
                if int(box.cls[0]) == 0:  # person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    if confidence > 0.5:  # confidence threshold
                        return (int(x1), int(y1), int(x2), int(y2)), confidence
    return None, 0.0


def draw_person_box(frame, bbox, confidence):
    """Draw bounding box around detected person"""
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Person: {confidence:.2f}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def draw_height_scale(frame, height_cm, bbox=None):
    """Draw a visual scale/ruler showing the jump height measurement"""
    frame_height, frame_width = frame.shape[:2]
    
    if bbox is None:
        # Use default position if no bounding box
        x1, y1 = 50, 50
        scale_width = 200  # Much wider scale
        scale_height = int(height_cm * 8)  # 8 pixels per cm for better visibility
    else:
        # Position scale next to the person
        px1, py1, px2, py2 = bbox
        x1 = px2 + 30  # 30 pixels to the right of person
        y1 = py1
        scale_width = 200  # Much wider scale
        scale_height = int(height_cm * 8)  # 8 pixels per cm for better visibility
    
    x2 = x1 + scale_width
    y2 = y1 + scale_height
    
    # Draw background rectangle for better visibility
    cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (255, 255, 255), 2)
    
    # Draw the main scale bar
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
    
    # Draw tick marks every 5cm with better visibility
    for i in range(0, int(height_cm) + 5, 5):
        tick_y = y2 - (i * 8)  # Convert cm to pixels (8 pixels per cm)
        if tick_y >= y1:
            # Draw thick tick mark
            cv2.line(frame, (x1, tick_y), (x1 - 20, tick_y), (255, 255, 255), 3)
            # Draw label with larger font
            cv2.putText(frame, f'{i}cm', (x1 - 120, tick_y + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw the actual jump height marker with thick line
    jump_height_pixels = int(height_cm * 8)
    marker_y = y2 - jump_height_pixels
    cv2.line(frame, (x1, marker_y), (x2, marker_y), (0, 255, 0), 5)
    
    # Draw jump height label with larger font
    cv2.putText(frame, f'JUMP: {height_cm:.1f}cm', (x2 + 15, marker_y + 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
    
    # Draw scale title with larger font
    cv2.putText(frame, 'HEIGHT SCALE', (x1, y1 - 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add a visual comparison - draw a person silhouette for scale
    person_height = 170  # Average person height in cm
    person_pixels = int(person_height * 8 / 10)  # Scale down for display
    person_y = y2 - person_pixels
    cv2.line(frame, (x1 + scale_width//2, person_y), (x1 + scale_width//2, y2), (100, 100, 255), 3)
    cv2.putText(frame, f'Person ({person_height}cm)', (x2 + 15, person_y + 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)


def draw_height_info(frame, height, duration, velocity, frame_time):
    """Draw jump height information on the frame"""
    # Create a semi-transparent overlay for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw only jump height with larger font
    cv2.putText(frame, f'Jump Height: {height:.2f} cm', (20, 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)


def draw_points(img, keypoints, scores):
    for idx in ankle_indices:
        if len(keypoints[0]) > idx and scores[0][idx] > 0.5:
            x, y = int(keypoints[0][idx][0]), int(keypoints[0][idx][1])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Draw a red dot to mark the ankle joint


def video_parser_with_yolo(video_path: str, output_path: str = None):
    """Parse video with YOLO person detection and pose estimation"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[video_parser] {total_frames} frames @ {fps:.1f} FPS")
    print(f"[video_parser] Resolution: {width}x{height}")

    # Setup video writer if output path is provided
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ankle_y_coords = []
    frame_times = []
    time_elapsed = 0
    
    # Store measurements for display
    final_height = 0
    final_duration = 0
    final_velocity = 0

    # First pass: collect all ankle data
    print("First pass: Collecting ankle data...")
    for _ in tqdm(range(total_frames), desc="Collecting data"):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect person with YOLO
        bbox, confidence = detect_person(frame)
        
        # if bbox is not None:
        #     # Extract person region for pose estimation
        #     x1, y1, x2, y2 = bbox
        #     person_region = frame[y1:y2, x1:x2]
            
            # if person_region.size > 0:  # Check if region is valid
        keypoints, scores = body(frame)
        
        # Adjust keypoint coordinates back to original frame
        # if len(keypoints[0]) > 0:
        left = keypoints[0][10] if len(keypoints[0])>10 else None
        right = keypoints[0][13] if len(keypoints[0])>13 else None
        if left is not None and right is not None:
            y = (left[1] + right[1]) / 2
            ankle_y_coords.append(frame.shape[0] - y)
            frame_times.append(time_elapsed)

        time_elapsed += 1/fps

    # Calculate measurements if we have data
    if len(ankle_y_coords) > 0:
        result = clean_frame(frame_times, ankle_y_coords)
        if result[0] is not None:
            x_data, y_data, filtered_times, filtered_heights = result
            
            # Fit the relationship between h and t based on a quadratic function
            coeffs, fit_times, fit_heights = fit_data(x_data, y_data, filtered_times)
            
            # Flight time
            final_duration = fit_times[-1] - fit_times[0]
            
            # Jumping height(cm) - using physics-based calculation
            final_height = cal_height(final_duration)
            
            # Scale calibration
            max_height = np.max(fit_heights)
            cm_per_pixel = final_height / max_height
            final_velocity = coeffs[1] * cm_per_pixel / 100

    # Second pass: create output video with measurements
    print("Second pass: Creating output video with measurements...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    time_elapsed = 0
    
    for _ in tqdm(range(total_frames), desc="Creating output video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect person with YOLO
        bbox, confidence = detect_person(frame)
        
        if bbox is not None:
            # Extract person region for pose estimation
            x1, y1, x2, y2 = bbox
            person_region = frame[y1:y2, x1:x2]
            
            if person_region.size > 0:  # Check if region is valid
                keypoints, scores = body(person_region)
                
                # Adjust keypoint coordinates back to original frame
                if len(keypoints[0]) > 0:
                    adjusted_keypoints = keypoints.copy()
                    for i in range(len(keypoints[0])):
                        if keypoints[0][i] is not None:
                            adjusted_keypoints[0][i][0] += x1
                            adjusted_keypoints[0][i][1] += y1
                    
                    # Draw ankle points
                    draw_points(frame, adjusted_keypoints, scores)
        
        # Draw person bounding box
        draw_person_box(frame, bbox, confidence)
        
        # Draw height information on frame
        draw_height_info(frame, final_height, final_duration, final_velocity, time_elapsed)
        
        # Write frame to output video
        if video_writer:
            video_writer.write(frame)

        time_elapsed += 1/fps

    cap.release()
    if video_writer:
        video_writer.release()
    
    return frame_times, ankle_y_coords


def clean_frame(frame_times, ankle_y_coords):
    # Data cleaning: identifying jump intervals
    if len(ankle_y_coords) < 2:
        return None, None, None, None
    
    y_diff = np.diff(ankle_y_coords)  # Calculate the change in adjacent points
    peak_diff = np.max(y_diff)
    thr = peak_diff * 0.5
    jump_start = np.where(y_diff > thr)[0][0]  # Find the first obvious point of increase
    jump_end = np.where(ankle_y_coords[jump_start:] < ankle_y_coords[jump_start])[0][0] + jump_start

    filtered_times = frame_times[jump_start:jump_end]
    filtered_heights = ankle_y_coords[jump_start:jump_end]

    # The height of the initial point
    initial_ankle_y = filtered_heights[0]
    # Relative height
    filtered_heights = [y - initial_ankle_y for y in filtered_heights]

    x_data = np.array(filtered_times) - filtered_times[0]
    y_data = np.array(filtered_heights)

    return x_data, y_data, filtered_times, filtered_heights


def fit_data(x_data, y_data, filtered_times):
    # Quadratic polynomial fitting
    coeffs = np.polyfit(x_data, y_data, 2)

    fit_times = np.linspace(filtered_times[0], filtered_times[-1], 100)
    fit_heights = np.polyval(coeffs, fit_times - filtered_times[0])

    return coeffs, fit_times, fit_heights


def debug_plot(filtered_times, filtered_heights, fit_times, fit_heights, height, duration, coeffs, cm_per_pixel):
    plt.figure(figsize=(10, 5))
    plt.scatter(filtered_times, filtered_heights, color='b', label='Cleaned Ankle Data')
    plt.plot(fit_times, fit_heights, color='r', linestyle='-', label=f'Polyfit: h = {coeffs[0]*cm_per_pixel:.2f} t^2 + {coeffs[1]*cm_per_pixel:.2f} t + {coeffs[2]*cm_per_pixel:.2f}')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Height(cm)')
    plt.title('Jump Motion Polynomial Fitting')
    plt.legend()
    plt.grid()
    plt.text(0.01, 0.95, f"Jump Height: {height:.2f} cm",
             fontsize=12, ha='left', va='top', color='red',
             transform=plt.gca().transAxes)

    plt.text(0.01, 0.90, f"Flight Time: {duration:.2f} s",
             fontsize=12, ha='left', va='top', color='green',
             transform=plt.gca().transAxes)

    plt.text(0.01, 0.85, f"Max velocity: {coeffs[1] * cm_per_pixel / 100:.2f} m/s",
             fontsize=12, ha='left', va='top', color='blue',
             transform=plt.gca().transAxes)
    plt.savefig('jump_plot_yolo.png', dpi=150) 