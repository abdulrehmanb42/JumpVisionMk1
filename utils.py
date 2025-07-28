import cv2
import numpy as np
from rtmlib import Body
from tqdm import tqdm
import matplotlib.pyplot as plt


openpose_skeleton = True  # True for openpose-style, False for mmpose-style
device = 'mps'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
ankle_indices = [10, 13]
body = Body(to_openpose=openpose_skeleton, mode='balanced', backend=backend, device=device)

g = 9.810665  # m/s²


def free_fall(t, h0):
    return 0.5 * g * t ** 2 + h0


def cal_height(t):
    return (0.5 * g * (t / 2) ** 2) * 100


def draw_points(img, keypoints, scores):
    for idx in ankle_indices:
        if len(keypoints[0]) > idx and scores[0][idx] > 0.5:
            x, y = int(keypoints[0][idx][0]), int(keypoints[0][idx][1])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Draw a red dot to mark the ankle joint


def video_parser(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video_parser] {total_frames} frames @ {fps:.1f} FPS")

    ankle_y_coords = []
    frame_times = []
    time_elapsed = 0

    # Iterate exactly once per frame and show progress
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, scores = body(frame)
        # (… your draw_points / ankle extraction …)
        # e.g.:
        left = keypoints[0][10] if len(keypoints[0])>10 else None
        right = keypoints[0][13] if len(keypoints[0])>13 else None
        if left is not None and right is not None:
            y = (left[1] + right[1]) / 2
            ankle_y_coords.append(frame.shape[0] - y)
            frame_times.append(time_elapsed)

        time_elapsed += 1/fps

    cap.release()
    return frame_times, ankle_y_coords


def clean_frame(frame_times, ankle_y_coords):
    # Data cleaning: identifying jump intervals
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
    plt.savefig('jump_plot.png', dpi=150)