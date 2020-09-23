import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os
from prettytable import PrettyTable
from colorama import Fore, Back, Style
from colorama import init
from utils import count_number_fps
from prettytable import PrettyTable
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Boundaries of different size classes of obstacles
OBSTACLE_SIZE_CLASSES = [5*5, 15*15, 30*30, 50*50, 100*100, 200*200]
OBSTACLE_TYPE_CLASSES = ['swimmer', 'vessel', 'other']

RESULTS_PATH = './results'


def get_arguments():
    """ Parse all the arguments provided from the CLI
    Returns: A list of parsed arguments
    """
    parser = argparse.ArgumentParser(description='Marine Obstacle Detection Benchmark - Evaluation statistics')
    parser.add_argument("--results-path", type=str, default=RESULTS_PATH,
                        help="Absolute path to the folder where the evaluation results are stored.")
    parser.add_argument("--method-name", type=str, required=True,
                        help="<Required> Name of the method to be analysed.")

    return parser.parse_args()


# Get evaluation statistics of the method
def main():
    init()  # Initialize colorama for colored output

    args = get_arguments()

    # Read results JSON file
    with open(os.path.join(args.results_path, 'results_%s.json' % args.method_name)) as f:
        results = json.load(f)
        
    # Read overlap results JSON file
    with open(os.path.join(args.results_path, 'results_%s_overlap.json' % args.method_name)) as f:
        overlap_results = json.load(f)

    # Get number of all sequences
    num_sequences = len(results['sequences'])

    # Initialize detection counters for different obstacle sizes (SIZES x 3(TP,FP,FN))
    det_sizes = np.zeros( (len(OBSTACLE_SIZE_CLASSES) + 1, 3) )
    det_sizes_danger = np.zeros( (len(OBSTACLE_SIZE_CLASSES) + 1, 3) )
    # Initialize detection counters for different obstacle types (TYPES x 2(TP,FN))
    det_types = np.zeros( (len(OBSTACLE_TYPE_CLASSES), 2) )
    det_types_danger = np.zeros( (len(OBSTACLE_SIZE_CLASSES) + 1, 3) )
    # Initialize detections by sequences (NUM SEQUENCES x 3(TP, FP, FN))
    det_sequences = np.zeros( (num_sequences, 3) )
    det_sequences_danger = np.zeros( (num_sequences, 3) )
    # Initialize water edge error for each sequence (NUM SEQUENCES x 3(TP, FP, FN))
    est_water_edge = np.zeros( (num_sequences, 3) )

    # Parse results
    for seq_id in range(num_sequences):
        # Check if the current sequence was evaluated
        if results['sequences'][seq_id]['evaluated']:
            tmp_rmse = np.zeros( (3, 1) )

            num_frames_in_sequence = len(results['sequences'][seq_id]['frames'])
            for frm in range(num_frames_in_sequence):
                # Update detections by sizes
                det_sizes = update_detection_by_sizes(results['sequences'][seq_id]['frames'][frm]['obstacles']['tp_list'],
                                                      0, det_sizes)
                det_sizes = update_detection_by_sizes(results['sequences'][seq_id]['frames'][frm]['obstacles']['fp_list'],
                                                      1, det_sizes)
                det_sizes = update_detection_by_sizes(results['sequences'][seq_id]['frames'][frm]['obstacles']['fn_list'],
                                                      2, det_sizes)

                det_sizes_danger = update_detection_by_sizes(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['tp_list'],
                                                             0, det_sizes_danger)
                det_sizes_danger = update_detection_by_sizes(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fp_list'],
                                                             1, det_sizes_danger)
                det_sizes_danger = update_detection_by_sizes(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fn_list'],
                                                             2, det_sizes_danger)

                # Update detections by type
                det_types = update_detection_by_types(results['sequences'][seq_id]['frames'][frm]['obstacles']['tp_list'],
                                                      0, det_types)
                det_types = update_detection_by_types(results['sequences'][seq_id]['frames'][frm]['obstacles']['fn_list'],
                                                      1, det_types)

                det_types_danger = update_detection_by_types(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['tp_list'],
                                                             0, det_types_danger)
                det_types_danger = update_detection_by_types(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fn_list'],
                                                             1, det_types_danger)

                # Update detections by sequence
                det_sequences[seq_id, 0] += len(results['sequences'][seq_id]['frames'][frm]['obstacles']['tp_list'])
                det_sequences[seq_id, 1] += count_number_fps(results['sequences'][seq_id]['frames'][frm]['obstacles']['fp_list'])
                det_sequences[seq_id, 2] += len(results['sequences'][seq_id]['frames'][frm]['obstacles']['fn_list'])

                det_sequences_danger[seq_id, 0] += len(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['tp_list'])
                det_sequences_danger[seq_id, 1] += count_number_fps(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fp_list'])
                det_sequences_danger[seq_id, 2] += len(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fn_list'])

                # Update water edge estimation
                tmp_rmse[0] += results['sequences'][seq_id]['frames'][frm]['rmse_t']
                tmp_rmse[1] += results['sequences'][seq_id]['frames'][frm]['rmse_o']
                tmp_rmse[2] += results['sequences'][seq_id]['frames'][frm]['rmse_u']

            est_water_edge[seq_id, 0] = tmp_rmse[0] / num_frames_in_sequence
            est_water_edge[seq_id, 1] = tmp_rmse[1] / num_frames_in_sequence
            est_water_edge[seq_id, 2] = tmp_rmse[2] / num_frames_in_sequence

    # Plot sizes detection rate
    fig = plt.figure(1, figsize=(15, 10))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.5)
    x_labels = ['TP', 'FP', 'FN']
    x_axis = np.arange(len(x_labels))
    maximum_number_of_detections = int(np.ceil(np.max(det_sizes) / 10.0)) * 10
    for i in range(1, len(OBSTACLE_SIZE_CLASSES)+1):
        if i == 0:
            area_min = 0
            area_max = OBSTACLE_SIZE_CLASSES[i]
        elif i == len(OBSTACLE_SIZE_CLASSES):
            area_min = OBSTACLE_SIZE_CLASSES[i-1]
            area_max = np.inf
        else:
            area_min = OBSTACLE_SIZE_CLASSES[i-1]
            area_max = OBSTACLE_SIZE_CLASSES[i]

        tmp_ax = plt.subplot(4, len(OBSTACLE_SIZE_CLASSES), i)
        tmp_ax.bar(x_axis, det_sizes[i, :])
        tmp_ax.bar(x_axis, det_sizes_danger[i, :])
        if i == 0:
            tmp_ax.set_ylabel('Number of detections')
        tmp_ax.set_xticks(x_axis)
        tmp_ax.set_xticklabels(x_labels)
        tmp_ax.set_title('[$%.f^2$, $%.f^2$)' % (np.sqrt(area_min), np.sqrt(area_max)))
        tmp_ax.set_ylim(bottom=0, top=maximum_number_of_detections)

    # Plot type detection rate
    labels = 'TP', 'FN'
    for i in range(len(OBSTACLE_TYPE_CLASSES)):
        if (det_types[i, 1] + det_types[i, 0]) == 0:
            percentage_tps = 1
            percentage_fns = 0
        else:
            percentage_tps = np.round((det_types[i, 0] / (det_types[i, 0] + det_types[i, 1]))*100)
            percentage_fns = 100 - percentage_tps

        if (det_types_danger[i, 1] + det_types_danger[i, 0]) == 0:
            percentage_tps_danger = 1
            percentage_fns_danger = 0
        else:
            percentage_tps_danger = np.round((det_types_danger[i, 0] /
                                              (det_types_danger[i, 0] + det_types_danger[i, 1])) * 100)
            percentage_fns_danger = 100 - percentage_tps_danger

        detection_percentages = [percentage_tps, percentage_fns]
        detection_percentages_danger = [percentage_tps_danger, percentage_fns_danger]

        explode = (0.1, 0)  # Only explode slice belonging to the TPs
        tmp_ax = plt.subplot(4, 3, 3 + (i + 1))
        tmp_ax.pie(detection_percentages, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        tmp_ax.axis('equal')
        tmp_ax.set_title('Detections of %s' % OBSTACLE_TYPE_CLASSES[i])

        tmp_ax = plt.subplot(4, 6, 21 + (i + 1))
        tmp_ax.pie(detection_percentages_danger, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True,
                   startangle=90)
        tmp_ax.axis('equal')
        tmp_ax.set_title('Detections of %s\n inside the danger zone' % OBSTACLE_TYPE_CLASSES[i])

    # Plot detections by sequence
    x_axis_sequences = np.arange(1, num_sequences+1)
    tmp_ax2 = plt.subplot(4, 2, 5)
    tmp_ax2.plot(x_axis_sequences, det_sequences[:, 0], marker='', color='olive', linewidth=2, label='TP')
    tmp_ax2.plot(x_axis_sequences, det_sequences[:, 1], marker='', color='orange', linewidth=2, linestyle='dashed',
                 label='FP')
    tmp_ax2.plot(x_axis_sequences, det_sequences[:, 2], marker='', color='red', linewidth=2, label='FN')
    tmp_ax2.set_ylabel('Number of detections')
    plt.title('Detections per sequences')
    tmp_ax2.legend()

    tmp_ax2_1 = plt.subplot(4, 2, 7)
    tmp_ax2_1.plot(x_axis_sequences, det_sequences_danger[:, 0], marker='', color='olive', linewidth=2, label='TP')
    tmp_ax2_1.plot(x_axis_sequences, det_sequences_danger[:, 1], marker='', color='orange', linewidth=2,
                   linestyle='dashed', label='FP')
    tmp_ax2_1.plot(x_axis_sequences, det_sequences_danger[:, 2], marker='', color='red', linewidth=2, label='FN')
    tmp_ax2_1.set_ylabel('Number of detections')
    plt.title('Detections per sequences within the danger zone')
    tmp_ax2_1.legend()

    # Plot water edge estimation by sequence
    x_seq_number = np.ones((num_sequences, 1))
    tmp_ax3 = plt.subplot(4, 2, 6)
    tmp_ax3.plot(x_axis_sequences, est_water_edge[:, 0], marker='', color='blue', linewidth=2, label='RMSE Total')
    tmp_ax3.plot(x_axis_sequences, est_water_edge[:, 1], marker='', color='purple', linewidth=2, label='RMSE Overshoot')
    tmp_ax3.plot(x_axis_sequences, est_water_edge[:, 2], marker='', color='pink', linewidth=2, label='RMSE Undershoot')
    # Average
    tmp_ax3.plot(x_axis_sequences, x_seq_number * np.mean(est_water_edge[:, 0]), marker='', color='blue', linewidth=1,
                 linestyle='dashed')  # , label='Average RMSE Total')
    tmp_ax3.plot(x_axis_sequences, x_seq_number * np.mean(est_water_edge[:, 1]), marker='', color='purple', linewidth=1,
                 linestyle='dashed')  # , label='Average RMSE Overshoot')
    tmp_ax3.plot(x_axis_sequences, x_seq_number * np.mean(est_water_edge[:, 2]), marker='', color='pink', linewidth=1,
                 linestyle='dashed')  # , label='Average RMSE Undershoot')
    plt.title('Water-edge estimation per sequences')
    tmp_ax3.set_ylabel('Water-Edge error [px]')
    tmp_ax3.legend()

    # Print detection overlap statistics...
    """
    plt.figure(2)
    plt.clf()

    overlap_perc_all = overlap_results['overlap_perc_all']
    overlap_perc_dng = overlap_results['overlap_perc_dng']
    

    x_d = np.linspace(0, 1, 100)
    kde = KernelDensity(bandwidth=0.05, kernel='gaussian')
    kde.fit(np.array(overlap_perc_all)[:, np.newaxis])
    kde.fit(np.array(overlap_perc_dng)[:, np.newaxis])

    # score_samples returns the log of the probability density
    logprob_all = kde.score_samples(x_d[:, None])
    max_density_all_ind = np.argmax(np.exp(logprob_all))
    max_density_all_val = np.exp(logprob_all[max_density_all_ind]) + 1
    
    logprob_dng = kde.score_samples(x_d[:, None])
    max_density_dng_ind = np.argmax(np.exp(logprob_dng))
    max_density_dng_val = np.exp(logprob_dng[max_density_dng_ind])
    
    current_overlap_threshold = float(results['parameters']['min_overlap'])

    # Plot graph
    plt.figure(2)
    plt.subplot(221)
    plt.fill_between(x_d, np.exp(logprob_all), alpha=0.5)
    plt.plot(overlap_perc_all, np.full_like(overlap_perc_all, -0.1), '|k', markeredgewidth=1)
    plt.scatter(x_d[max_density_all_ind], np.exp(logprob_all[max_density_all_ind]))
    plt.text(x_d[max_density_all_ind], np.exp(logprob_all[max_density_all_ind]), '%.02f' % x_d[max_density_all_ind])
    plt.plot([current_overlap_threshold, current_overlap_threshold], [-0.2, max_density_all_val], ':r')
    plt.ylim([-0.2, max_density_all_val])
    
    plt.subplot(222)
    tmp_hist_all_y, _, _ = plt.hist(overlap_perc_all, bins=10)
    max_hist_all_y = tmp_hist_all_y.max()
    plt.plot([current_overlap_threshold, current_overlap_threshold], [0, max_hist_all_y], ':r')
    plt.ylim([0, max_hist_all_y])
    plt.xlim([0, 1])
    
    plt.subplot(223)
    plt.fill_between(x_d, np.exp(logprob_dng), alpha=0.5)
    plt.plot(overlap_perc_dng, np.full_like(overlap_perc_dng, -0.1), '|k', markeredgewidth=1)
    plt.scatter(x_d[max_density_dng_ind], np.exp(logprob_dng[max_density_dng_ind]))
    plt.text(x_d[max_density_dng_ind], np.exp(logprob_dng[max_density_dng_ind]), '%.02f' % x_d[max_density_dng_ind])
    plt.plot([current_overlap_threshold, current_overlap_threshold], [-0.2, max_density_all_val], ':r')
    plt.ylim([-0.2, max_density_all_val])
    
    plt.subplot(224)
    plt.hist(overlap_perc_dng, bins=10)
    plt.plot([current_overlap_threshold, current_overlap_threshold], [0, max_hist_all_y], ':r')
    plt.ylim([0, max_hist_all_y])
    plt.xlim([0, 1])
    """

    # Print brief statistics
    table = PrettyTable()

    tmp_edge = np.ceil(np.mean(est_water_edge[:, 0]))
    tmp_oshot = np.mean(est_water_edge[:, 1]) / (np.mean(est_water_edge[:, 1]) +
                                                 np.mean(est_water_edge[:, 2])) * 100
    tmp_ushot = np.mean(est_water_edge[:, 2]) / (np.mean(est_water_edge[:, 1]) +
                                                 np.mean(est_water_edge[:, 2])) * 100

    wedge_line = '%d px ' + Fore.LIGHTRED_EX + '(+%.01f%%, ' + Fore.LIGHTYELLOW_EX + '-%.01f%%)' + Fore.WHITE
    wedge_line = wedge_line % (tmp_edge, tmp_oshot, tmp_ushot)

    tp_line = Fore.LIGHTGREEN_EX + '%d (%d)' + Fore.WHITE
    tp_line = tp_line % (np.sum(det_sequences[:, 0]), np.sum(det_sequences_danger[:, 0]))

    fp_line = Fore.LIGHTYELLOW_EX + '%d (%d)' + Fore.WHITE
    fp_line = fp_line % (np.sum(det_sequences[:, 1]), np.sum(det_sequences_danger[:, 1]))

    fn_line = Fore.LIGHTRED_EX + '%d (%d)' + Fore.WHITE
    fn_line = fn_line % (np.sum(det_sequences[:, 2]), np.sum(det_sequences_danger[:, 2]))

    f1_score = (2 * np.sum(det_sequences[:, 0])) / (2 * np.sum(det_sequences[:, 0]) + np.sum(det_sequences[:, 1]) + 
                                                    np.sum(det_sequences[:, 2])) * 100
    f1_score_d = (2 * np.sum(det_sequences_danger[:, 0])) / (2 * np.sum(det_sequences_danger[:, 0]) + 
                                                             np.sum(det_sequences_danger[:, 1]) +
                                                             np.sum(det_sequences_danger[:, 2])) * 100

    f1_line = '%.01f%% (%.01f%%)' % (np.mean(f1_score), 0)  #f1_score_d)

    table.field_names = ['Water-edge RMSE', 'TPs', 'FPs', 'FNs', 'F1']
    table.add_row([wedge_line, tp_line, fp_line, fn_line, f1_line])

    print(table.get_string(title="Results for method %s on %d sequence/s" % (args.method_name, num_sequences)))
    
    plt.show()


# Function parses through the list of detections and checks to which type class it belongs
def update_detection_by_types(det_list, type_index, det_types):
    # type_index: 0 = TP, 1 = FN
    num_detections = len(det_list)
    for i in range(num_detections):
        det_type = det_list[i]['type']
        if det_type.lower() == OBSTACLE_TYPE_CLASSES[0].lower():
            det_types[0, type_index] += 1
        elif det_type.lower() == OBSTACLE_TYPE_CLASSES[1].lower():
            det_types[1, type_index] += 1
        else:
            det_types[2, type_index] += 1

    return det_types


# Function parses through the list of detections and checks in which size class does the detection fall into
def update_detection_by_sizes(det_list, type_index, det_sizes):
    # type_index: 0 = TP, 1 = FP, 2 = FN
    num_detections = len(det_list)
    for i in range(num_detections):
        det_area = det_list[i]['area']
        for j in range(len(OBSTACLE_SIZE_CLASSES) + 1):
            if j == 0 and det_area <= OBSTACLE_SIZE_CLASSES[j]:
                if type_index == 1:
                    det_sizes[j, type_index] += det_list[i]['num_triggers']
                else:
                    det_sizes[j, type_index] += 1
            elif j == len(OBSTACLE_SIZE_CLASSES) and det_area > OBSTACLE_SIZE_CLASSES[j-1]:
                if type_index == 1:
                    det_sizes[j, type_index] += det_list[i]['num_triggers']
                else:
                    det_sizes[j, type_index] += 1
            elif OBSTACLE_SIZE_CLASSES[j - 1] < det_area <= OBSTACLE_SIZE_CLASSES[j]:
                if type_index == 1:
                    det_sizes[j, type_index] += det_list[i]['num_triggers']
                else:
                    det_sizes[j, type_index] += 1

    return det_sizes


if __name__ == '__main__':
    main()
