import numpy as np
import argparse
import time
import torch
from utils_yolo import video_parser_with_yolo, clean_frame, fit_data, cal_height, debug_plot

def main(args):
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Start timing
    start_time = time.time()
    
    # Parse key points with YOLO person detection
    frame_times, ankle_y_coords = video_parser_with_yolo(args.input, args.output)
    
    if len(ankle_y_coords) == 0:
        print("No person detected or no valid ankle data found!")
        return
    
    # Extract the start and end frames of the jump
    result = clean_frame(frame_times, ankle_y_coords)
    if result[0] is None:
        print("No valid jump detected!")
        return
    
    x_data, y_data, filtered_times, filtered_heights = result
    
    # Fit the relationship between h and t based on a quadratic function
    coeffs, fit_times, fit_heights = fit_data(x_data, y_data, filtered_times)
    
    # Flight time
    duration = fit_times[-1] - fit_times[0]
    
    # Jumping height(cm) - using physics-based calculation
    height = cal_height(duration)
    
    # Scale calibration
    max_height = np.max(fit_heights)
    cm_per_pixel = height / max_height
    filtered_heights = [y * cm_per_pixel for y in filtered_heights]
    fit_heights = [y * cm_per_pixel for y in fit_heights]
    
    # Draw a comparison chart if debug mode is enabled
    if args.debug:
        debug_plot(
            filtered_times,
            filtered_heights,
            fit_times,
            fit_heights,
            height,
            duration,
            coeffs,
            cm_per_pixel
        )

    # End timing
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(
        f'\nJump Height: {height:.2f} cm',
        f'\nFlight Time: {duration:.2f} s',
        f'\nMax velocity: {coeffs[1]*cm_per_pixel/100:.2f} m/s',
        f'\nProcessing Time: {processing_time:.2f} s'
    )
    
    if torch.cuda.is_available():
        print(f'GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jump height measurement with YOLO person detection (GPU optimized)')

    parser.add_argument(
        'input',
        help='path to input video'
    )

    parser.add_argument(
        '-o', '--output',
        help='path to output video with bounding boxes and measurements',
        default=None
    )

    parser.add_argument(
        '-d', '--debug',
        help='Toggle debug mode to generate plot',
        action='store_true',
        default=False
    )

    args = parser.parse_args()

    main(args) 