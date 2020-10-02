import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

RESULTS_PATH = './results'
NUM_SEQUENCES = 94


def get_arguments():
    """ Parse all the arguments provided from the CLI
    Returns: A list of parsed arguments
    """
    parser = argparse.ArgumentParser(description='Marine Obstacle Detection Benchmark.')
    parser.add_argument("--results-path", type=str, default=RESULTS_PATH,
                        help="Absolute path to the folder where evaluation results are stored.")
    parser.add_argument("--methods", type=str, nargs='+', required=True,
                        help="<Required> First method name. This should be equal to the folder name in which the "
                             "segmentation masks are located.")

    return parser.parse_args()


def main():
    args = get_arguments()

    # Get number of methods
    num_methods = len(args.methods)

    # Number of parameters to visualize
    N = 6

    # Initialize array of size (num_methods x 8) for storing detections of each method
    # obstacle TP, obstacle FP, obstacle FN, danger TP, danger FP, danger FN, land TP, land FN
    total_detections = np.zeros((num_methods, 8))
    total_detections_per_sequence = np.zeros((num_methods, 8))

    # Initialize array of size (num_methods x 94) for storing water-edge rmse of each method and each sequence
    water_edges = np.zeros((num_methods, NUM_SEQUENCES))
    
    method_names = []

    # Detection rates (per sequence)
    detection_rates = np.zeros((num_methods, 6))
    f1_per_sequence = np.zeros((num_methods, NUM_SEQUENCES))

    # Loop through the methods...
    for i in range(num_methods):
        # Get and append results
        # Load results
        with open(os.path.join(args.results_path, 'results_%s.json' % args.methods[i])) as f:
            tmp_results = json.load(f)
            total_detections, water_edges = get_detection_data(tmp_results, total_detections, water_edges, i)
            detection_rates, f1_per_sequence = get_detection_data_per_sequence(tmp_results, detection_rates,
                                                                               f1_per_sequence, i)
        
            # Add method name
            method_names.append(tmp_results['method_name'])

    # Initialize spider-plot
    theta = radar_factory(N, frame='polygon')
    detection_rates *= 100 / NUM_SEQUENCES

    # Generate detection data statistics
    data = generate_detection_data(total_detections)
    # data_per_sequence = generate_detection_data(total_detections_per_sequence)
    # _ = data_per_sequence.pop(0)
    spoke_labels = data.pop(0)

    # FIGURE 1 - ALL
    fig, axes = plt.subplots(figsize=(9, 9), nrows=1, ncols=2)
    axes[0].axis('off')
    axes[1].axis('off')

    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:pink', 'tab:olive']  # Colors for top-five performing methods

    print(data)

    ax1 = fig.add_subplot(1, 2, 1, projection='radar')
    # Loop through the results of the methods and plot them in spiderplot
    for d, color in zip(data[0][1], colors):
        ax1.plot(theta, d, color=color)
        ax1.fill(theta, d, facecolor=color, alpha=0.25)
    ax1.set_varlabels(spoke_labels)

    # Add legend
    ax1.legend(method_names, loc=(0.9, .95), labelspacing=0.1, fontsize='small')

    # Water edge comparison
    ax2 = fig.add_subplot(1, 2, 2, projection='rectilinear')
    x_axis = np.arange(1, NUM_SEQUENCES+1, 1)
    ax2.set_xlim(1, NUM_SEQUENCES)
    # ax2.set_xticks(x_axis)
    for d, color in zip(water_edges, colors):
        ax2.plot(x_axis, d, color=color)

    #plt.show()


    # FIGURE 2 - RESULTS PER SEQUENCE
    fig2, axes2 = plt.subplots(figsize=(9, 9), nrows=1, ncols=2)
    axes2[0].axis('off')
    axes2[1].axis('off')

    fig2.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    print(detection_rates)

    ax21 = fig2.add_subplot(1, 2, 1, projection='radar')
    # Loop through the results of the methods and plot them in spiderplot
    for d, color in zip(detection_rates, colors):
        ax21.plot(theta, d, color=color)
        ax21.fill(theta, d, facecolor=color, alpha=0.25)
    ax21.set_varlabels(spoke_labels)

    # Add legend
    ax21.legend(method_names, loc=(0.9, .95), labelspacing=0.1, fontsize='small')

    # Water edge comparison
    ax22 = fig2.add_subplot(2, 2, 2, projection='rectilinear')
    ax22.set_xlim(1, NUM_SEQUENCES)
    # ax2.set_xticks(x_axis)
    for d, color in zip(water_edges, colors):
        ax22.plot(x_axis, d, color=color)

    # F1 per sequence
    f1_per_sequence *= 100
    ax23 = fig2.add_subplot(2, 2, 4, projection='rectilinear')
    ax23.set_xlim(1, NUM_SEQUENCES)
    for d, color in zip(f1_per_sequence, colors):
        ax23.plot(x_axis, d, color=color)

    plt.show()


def generate_detection_data(detections_data):
    # Calculate needed data
    num_methods = detections_data.shape[0]
    processed_data = np.zeros((num_methods, 6))

    for i in range(num_methods):
        processed_data[i, 0] = detections_data[i, 0] / (detections_data[i, 0] + detections_data[i, 2])
        processed_data[i, 1] = detections_data[i, 2] / (detections_data[i, 0] + detections_data[i, 2])
        processed_data[i, 2] = detections_data[i, 1] / (detections_data[i, 0] + detections_data[i, 1]) 
        processed_data[i, 3] = (2 * detections_data[i, 0]) / (2 * detections_data[i, 0] + detections_data[i, 1] + 
                                                              detections_data[i, 2])
        processed_data[i, 4] = (2 * detections_data[i, 3]) / (2 * detections_data[i, 3] + detections_data[i, 4] + 
                                                              detections_data[i, 5])
        processed_data[i, 5] = detections_data[i, 6] / (detections_data[i, 6] + detections_data[i, 7])
        
    processed_data = np.nan_to_num(processed_data) * 100

    # Generate data structure
    data = [
        ['TP Rate', 'FN Rate', 'False Discovery Rate', 'F1-score', 'F1-score within danger', 'Land detection rate'],
        ('Quantitative comparison', processed_data)
    ]
    return data


def get_detection_data_per_sequence(results, detection_rates, f1_per_sequence, i):
    # Get the number of all sequences
    num_sequences = len(results['sequences'])

    # Parse results
    for seq_id in range(num_sequences):
        # Check if the current sequence was evaluated
        if results['sequences'][seq_id]['evaluated']:
            # Get the number of frames in the sequence
            num_frames_in_sequence = len(results['sequences'][seq_id]['frames'])

            # Loop through the frames
            tmp_detections = np.zeros(8)
            for frm in range(num_frames_in_sequence):
                # Update the number of obstacles detected (whole image)
                tmp_detections[0] += len(results['sequences'][seq_id]['frames'][frm]['obstacles']['tp_list'])
                tmp_detections[1] += len(results['sequences'][seq_id]['frames'][frm]['obstacles']['fp_list'])
                tmp_detections[2] += len(results['sequences'][seq_id]['frames'][frm]['obstacles']['fn_list'])

                # Update the number of obstacles detected inside the danger zone
                tmp_detections[3] += len(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['tp_list'])
                tmp_detections[4] += len(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fp_list'])
                tmp_detections[5] += len(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fn_list'])

                # Update land detections
                tmp_detections[6] += results['sequences'][seq_id]['frames'][frm]['we_detections'][0]
                tmp_detections[7] += results['sequences'][seq_id]['frames'][frm]['we_detections'][1]

            if tmp_detections[0] + tmp_detections[2] > 0:
                # TP-rate
                detection_rates[i, 0] += tmp_detections[0] / (tmp_detections[0] + tmp_detections[2])
                # FN-rate
                detection_rates[i, 1] += tmp_detections[2] / (tmp_detections[0] + tmp_detections[2])

            if tmp_detections[0] + tmp_detections[1] > 0:
                # False-Discovery Rate
                detection_rates[i, 2] += tmp_detections[1] / (tmp_detections[0] + tmp_detections[1])

            if tmp_detections[0] + tmp_detections[1] + tmp_detections[2] > 0:
                # F1 score
                detection_rates[i, 3] += (2 * tmp_detections[0]) / (2 * tmp_detections[0] + tmp_detections[1] +
                                                                    tmp_detections[2])
                f1_per_sequence[i, seq_id] = (2 * tmp_detections[0]) / (2 * tmp_detections[0] + tmp_detections[1] +
                                                                        tmp_detections[2])

            if tmp_detections[3] + tmp_detections[4] + tmp_detections[5] > 0:
                # F1 danger zone
                detection_rates[i, 4] += (2 * tmp_detections[3]) / (2 * tmp_detections[3] + tmp_detections[4] +
                                                                    tmp_detections[5])

            # Water edge
            if tmp_detections[6] + tmp_detections[7] > 0:
                detection_rates[i, 5] += tmp_detections[6] / (tmp_detections[6] + tmp_detections[7])

    return detection_rates, f1_per_sequence


def get_detection_data(results, total_detections, water_edges, method_num):
    # Get number of all sequences
    num_sequences = len(results['sequences'])

    # Parse results
    for seq_id in range(num_sequences):
        # Check if the current sequence was evaluated
        if results['sequences'][seq_id]['evaluated']:
            # Get number of frames in sequence
            num_frames_in_sequence = len(results['sequences'][seq_id]['frames'])

            # Loop through the frames
            for frm in range(num_frames_in_sequence):
                # Update number of obstacle detection
                total_detections[method_num, 0] += len(results['sequences'][seq_id]['frames'][frm]['obstacles']['tp_list'])  # TPs
                total_detections[method_num, 1] += len(results['sequences'][seq_id]['frames'][frm]['obstacles']['fp_list'])  # FPs
                total_detections[method_num, 2] += len(results['sequences'][seq_id]['frames'][frm]['obstacles']['fn_list'])  # FNs

                # Update number of obstacle detection within the danger zone
                total_detections[method_num, 3] += len(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['tp_list'])  # TPs
                total_detections[method_num, 4] += len(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fp_list'])  # FPs
                total_detections[method_num, 5] += len(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fn_list'])  # FNs

                # Update land detections
                tmp_land_detections = results['sequences'][seq_id]['frames'][frm]['we_detections']  # Land detections
                total_detections[method_num, 6] += tmp_land_detections[0]  # TPs land
                total_detections[method_num, 7] += tmp_land_detections[1]  # FNs land

                water_edges[method_num, seq_id] += results['sequences'][seq_id]['frames'][frm]['we_rmse']

            water_edges[method_num, seq_id] /= num_frames_in_sequence

    return total_detections, water_edges


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


if __name__ == '__main__':
    main()
