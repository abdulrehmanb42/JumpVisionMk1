import numpy as np
import argparse
from utils import video_parser, clean_frame, fit_data, cal_height, debug_plot

def main(args):
    # parse key points
    frame_times, ankle_y_coords = video_parser(args.input)
    # Extract the start and end frames of the jump
    x_data, y_data, filtered_times, filtered_heights = clean_frame(frame_times, ankle_y_coords)
    # Fit the relationship between h and t based on a quadratic function
    coeffs, fit_times, fit_heights = fit_data(x_data, y_data, filtered_times)
    # Flight time
    duration = fit_times[-1]-fit_times[0]
    # Jumping height(cm)
    height = cal_height(duration)
    max_height = np.max(fit_heights)
    cm_per_pixel = height / max_height
    filtered_heights = [y * cm_per_pixel for y in filtered_heights]
    fit_heights = [y * cm_per_pixel for y in fit_heights]    # Draw a comparison chart
    
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

    print(
        f'\nJump Height: {height:.2f} cm',
        f'\nFlight Time: {duration:.2f} s',
        f'\nMax velocity: {coeffs[1]*cm_per_pixel/100:.2f} m/s'
    )
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input',
        help = 'path to video'
    )

    parser.add_argument(
        '-d', '--debug',
        help = 'Toggle debug mode',
        default = False
    )

    args = parser.parse_args()

    main(args)

