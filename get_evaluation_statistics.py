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
from configs import get_cfg

import matplotlib
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype']  = 42


def get_arguments():
    """ Parse all the arguments provided from the CLI
    Returns: A list of parsed arguments
    """
    parser = argparse.ArgumentParser(description='Marine Obstacle Detection Benchmark - Evaluation statistics')
    parser.add_argument("method", type=str,
                        help="<Required> Name of the method to be analysed.")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Config file to use. If not specified, the default config is used.")

    return parser.parse_args()


# Get evaluation statistics of the method
def main():
    init()  # Initialize colorama for colored output

    args = get_arguments()
    cfg  = get_cfg(args)

    # Set font sizes for figures
    matplotlib.rcParams['pdf.fonttype'] = cfg.VISUALIZATION.FONT_SIZE
    matplotlib.rcParams['ps.fonttype']  = cfg.VISUALIZATION.FONT_SIZE

    # Read results JSON file
    with open(os.path.join(cfg.PATHS.RESULTS, 'results_%s.json' % args.method)) as f:
        results = json.load(f)

    # Read overlap results JSON file
    #with open(os.path.join(args.results_path, 'results_%s_overlap.json' % args.method_name)) as f:
    #    overlap_results = json.load(f)

    # Get number of all sequences
    num_sequences = len(results['sequences'])

    # Initialize detection counters for different obstacle sizes (SIZES x 3(TP,FP,FN))
    det_sizes = np.zeros((len(cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES) + 1, 3))
    det_sizes_danger = np.zeros((len(cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES) + 1, 3))
    # Initialize detection counters for different obstacle types (TYPES x 2(TP,FN))
    det_types = np.zeros((len(cfg.ANALYSIS.OBSTACLE_TYPE_CLASSES), 2))
    det_types_danger = np.zeros((len(cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES) + 1, 3))
    # Initialize detections by sequences (NUM SEQUENCES x 3(TP, FP, FN))
    det_sequences = np.zeros((num_sequences, 3))
    det_sequences_danger = np.zeros((num_sequences, 3))
    # Initialize water edge error for each sequence (NUM SEQUENCES x 3(TP, FP, FN))
    est_water_edge = np.zeros((num_sequences, 6))

    # Total number of frames in the dataset
    num_frames_total = 0

    # Parse results
    debug_all_detection = 0
    for seq_id in range(num_sequences):
        # Check if the current sequence was evaluated
        if results['sequences'][seq_id]['evaluated']:
            tmp_rmse = np.zeros((5, 1))

            num_frames_in_sequence = len(results['sequences'][seq_id]['frames'])
            num_frames_total      += num_frames_in_sequence

            for frm in range(num_frames_in_sequence):
                # Update detections by sizes
                det_sizes = update_detection_by_sizes(results['sequences'][seq_id]['frames'][frm]['obstacles']['tp_list'],
                                                      0, det_sizes, cfg)
                det_sizes = update_detection_by_sizes(results['sequences'][seq_id]['frames'][frm]['obstacles']['fp_list'],
                                                      1, det_sizes, cfg)
                det_sizes = update_detection_by_sizes(results['sequences'][seq_id]['frames'][frm]['obstacles']['fn_list'],
                                                      2, det_sizes, cfg)
                debug_all_detection += len(results['sequences'][seq_id]['frames'][frm]['obstacles']['tp_list'])

                det_sizes_danger = update_detection_by_sizes(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['tp_list'],
                                                             0, det_sizes_danger, cfg)
                det_sizes_danger = update_detection_by_sizes(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fp_list'],
                                                             1, det_sizes_danger, cfg)
                det_sizes_danger = update_detection_by_sizes(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fn_list'],
                                                             2, det_sizes_danger, cfg)

                # Update detections by type
                det_types = update_detection_by_types(results['sequences'][seq_id]['frames'][frm]['obstacles']['tp_list'],
                                                      0, det_types, cfg)
                det_types = update_detection_by_types(results['sequences'][seq_id]['frames'][frm]['obstacles']['fn_list'],
                                                      1, det_types, cfg)

                det_types_danger = update_detection_by_types(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['tp_list'],
                                                             0, det_types_danger, cfg)
                det_types_danger = update_detection_by_types(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fn_list'],
                                                             1, det_types_danger, cfg)

                # Update detections by sequence
                det_sequences[seq_id, 0] += len(results['sequences'][seq_id]['frames'][frm]['obstacles']['tp_list'])
                det_sequences[seq_id, 1] += count_number_fps(results['sequences'][seq_id]['frames'][frm]['obstacles']['fp_list'])
                det_sequences[seq_id, 2] += len(results['sequences'][seq_id]['frames'][frm]['obstacles']['fn_list'])

                det_sequences_danger[seq_id, 0] += len(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['tp_list'])
                det_sequences_danger[seq_id, 1] += count_number_fps(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fp_list'])
                det_sequences_danger[seq_id, 2] += len(results['sequences'][seq_id]['frames'][frm]['obstacles_danger']['fn_list'])

                # Update water edge estimation
                tmp_rmse[0] += results['sequences'][seq_id]['frames'][frm]['we_rmse']
                tmp_rmse[1] += results['sequences'][seq_id]['frames'][frm]['we_o']
                tmp_rmse[2] += results['sequences'][seq_id]['frames'][frm]['we_u']
                tmp_rmse[3] += results['sequences'][seq_id]['frames'][frm]['we_detections'][0]
                tmp_rmse[4] += results['sequences'][seq_id]['frames'][frm]['we_detections'][1]

            est_water_edge[seq_id, 0] = tmp_rmse[0]
            est_water_edge[seq_id, 1] = tmp_rmse[0] / num_frames_in_sequence
            est_water_edge[seq_id, 2] = tmp_rmse[1]
            est_water_edge[seq_id, 3] = tmp_rmse[2]
            est_water_edge[seq_id, 4] = tmp_rmse[3]
            est_water_edge[seq_id, 5] = tmp_rmse[4]

    # Plot sizes detection rate
    fig = plt.figure(1, figsize=(15, 10))
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.5)
    x_labels = ['TP', 'FP', 'FN']
    x_axis = np.arange(len(x_labels))
    maximum_number_of_detections = int(np.ceil(np.max(det_sizes) / 10.0)) * 10
    for i in range(1, len(cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES)+1):
        if i == 0:
            area_min = 0
            area_max = cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES[i]
        elif i == len(cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES):
            area_min = cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES[i-1]
            area_max = np.inf
        else:
            area_min = cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES[i-1]
            area_max = cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES[i]

        tmp_ax = plt.subplot(4, len(cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES), i)
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
    for i in range(len(cfg.ANALYSIS.OBSTACLE_TYPE_CLASSES)):
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
        tmp_ax.set_title('Detections of %s' % cfg.ANALYSIS.OBSTACLE_TYPE_CLASSES[i])

        tmp_ax = plt.subplot(4, 6, 21 + (i + 1))
        tmp_ax.pie(detection_percentages_danger, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True,
                   startangle=90)
        tmp_ax.axis('equal')
        tmp_ax.set_title('Detections of %s\n inside the danger zone' % cfg.ANALYSIS.OBSTACLE_TYPE_CLASSES[i])

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
    tmp_ax3.plot(x_axis_sequences, est_water_edge[:, 1], marker='', color='blue', linewidth=2, label='RMSE Total')
    # tmp_ax3.plot(x_axis_sequences, est_water_edge[:, 1], marker='', color='purple', linewidth=2, label='RMSE Overshoot')
    # tmp_ax3.plot(x_axis_sequences, est_water_edge[:, 2], marker='', color='pink', linewidth=2, label='RMSE Undershoot')
    # Average
    tmp_ax3.plot(x_axis_sequences, x_seq_number * np.mean(est_water_edge[:, 1]), marker='', color='blue', linewidth=1,
                 linestyle='dashed', label='Average RMSE')
    # tmp_ax3.plot(x_axis_sequences, x_seq_number * np.mean(est_water_edge[:, 1]), marker='', color='purple', linewidth=1,
    #              linestyle='dashed')  # , label='Average RMSE Overshoot')
    # tmp_ax3.plot(x_axis_sequences, x_seq_number * np.mean(est_water_edge[:, 2]), marker='', color='pink', linewidth=1,
    #              linestyle='dashed')  # , label='Average RMSE Undershoot')
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
    table_sizes = PrettyTable()
    table_sizes_danger = PrettyTable()
    table_ratios = PrettyTable()

    #tmp_edge = np.ceil(np.mean(est_water_edge[:, 1]))
    tmp_edge = np.sum(est_water_edge[:, 0]) / num_frames_total
    tmp_we_percentage = (np.sum(est_water_edge[:, 4]) / (np.sum(est_water_edge[:, 4]) + np.sum(est_water_edge[:, 5])))
    tmp_oshot = np.sum(est_water_edge[:, 2]) / (np.sum(est_water_edge[:, 2]) + np.sum(est_water_edge[:, 3]))
    tmp_ushot = np.sum(est_water_edge[:, 3]) / (np.sum(est_water_edge[:, 2]) + np.sum(est_water_edge[:, 3]))

    wedge_line = '%.1f px (%0.1f)' + Fore.LIGHTRED_EX + '(+%.01f%%, ' + Fore.LIGHTYELLOW_EX + '-%.01f%%)' + Fore.WHITE
    wedge_line = wedge_line % (tmp_edge, tmp_we_percentage * 100, tmp_oshot * 100, tmp_ushot * 100)

    tmp_tp_all = np.sum(det_sequences[:, 0])
    tmp_tp_dz = np.sum(det_sequences_danger[:, 0])
    tp_line = Fore.LIGHTGREEN_EX + '%d (%d)' + Fore.WHITE
    tp_line = tp_line % (tmp_tp_all, tmp_tp_dz)

    tmp_fp_all = np.sum(det_sequences[:, 1])
    tmp_fp_dz = np.sum(det_sequences_danger[:, 1])
    fp_line = Fore.LIGHTYELLOW_EX + '%d (%d)' + Fore.WHITE
    fp_line = fp_line % (tmp_fp_all, tmp_fp_dz)

    tmp_fn_all = np.sum(det_sequences[:, 2])
    tmp_fn_dz = np.sum(det_sequences_danger[:, 2])
    fn_line = Fore.LIGHTRED_EX + '%d (%d)' + Fore.WHITE
    fn_line = fn_line % (tmp_fn_all, tmp_fn_dz)

    f1_score = (2 * np.sum(det_sequences[:, 0])) / (2 * np.sum(det_sequences[:, 0]) + np.sum(det_sequences[:, 1]) +
                                                    np.sum(det_sequences[:, 2]))
    f1_score_d = (2 * np.sum(det_sequences_danger[:, 0])) / (2 * np.sum(det_sequences_danger[:, 0]) +
                                                             np.sum(det_sequences_danger[:, 1]) +
                                                             np.sum(det_sequences_danger[:, 2]))

    f1_line = '%.01f%% (%.01f%%)' % (f1_score * 100, f1_score_d * 100)

    table.field_names = ['Water-edge RMSE', 'TPs', 'FPs', 'FNs', 'F1']
    table.add_row([wedge_line, tp_line, fp_line, fn_line, f1_line])

    export_data = {
        'water-edge':{
            'error': tmp_edge,
            'oshot': tmp_oshot,
            'ushot': tmp_ushot,
            'robustness': tmp_we_percentage
        },
        'obstacles': {
            'overall': {
                'TP': tmp_tp_all,
                'FP': tmp_fp_all,
                'FN': tmp_fn_all,
                'F1': f1_score
            },
            'danger-zone': {
                'TP': tmp_tp_dz,
                'FP': tmp_fp_dz,
                'FN': tmp_fn_dz,
                'F1': f1_score_d
            }
        }
    }

    with open(os.path.join(cfg.PATHS.RESULTS, 'results_%s_summary.json' % args.method), 'w') as f:
        json.dump(export_data, f)

    print(table.get_string(title="Results for method %s on %d sequence/s" % (args.method, num_sequences)))

    tp_rate = np.full((6, 2), -1.)
    fp_rate = np.full((6, 2), -1.)
    ratios = np.zeros((8, 2), dtype=np.float64)
    print(det_sizes)
    for i in range(6):
        tmp = det_sizes[i+1, 0] + det_sizes[i+1, 2]
        if tmp > 0:
            tp_rate[i, 0] = 100 * det_sizes[i+1, 0] / (det_sizes[i+1, 0] + det_sizes[i+1, 2])

        tmp = det_sizes[i+1, 0] + det_sizes[i+1, 1]
        if tmp > 0:
            fp_rate[i, 0] = 100 * det_sizes[i+1, 1] / (det_sizes[i+1, 0] + det_sizes[i+1, 1])

        tmp = det_sizes_danger[i+1, 0] + det_sizes_danger[i+1, 2]
        if tmp > 0:
            tp_rate[i, 1] = 100 * det_sizes_danger[i+1, 0] / (det_sizes_danger[i+1, 0] + det_sizes_danger[i+1, 2])

        tmp = det_sizes_danger[i+1, 0] + det_sizes_danger[i+1, 1]
        if tmp > 0:
            fp_rate[i, 1] = 100 * det_sizes_danger[i+1, 1] / (det_sizes_danger[i+1, 0] + det_sizes_danger[i+1, 1])

        ratios[i, 0] = 100 * (det_sizes_danger[i+1, 0] + det_sizes_danger[i+1, 2]) / (det_sizes[i+1, 0] + det_sizes[i+1, 2])
        ratios[i, 1] = 100 * (det_sizes_danger[i+1, 1]) / (det_sizes[i+1, 1])

    # Detection sizes numbers
    table_sizes.field_names = ['tiny', 'very small', 'small', 'medium', 'large', 'very large']
    table_sizes.add_row(['%.01f' % tp_rate[0, 0],
                         '%.01f' % tp_rate[1, 0],
                         '%.01f' % tp_rate[2, 0],
                         '%.01f' % tp_rate[3, 0],
                         '%.01f' % tp_rate[4, 0],
                         '%.01f' % tp_rate[5, 0]])

    table_sizes.add_row(['%.01f' % fp_rate[0, 0],
                         '%.01f' % fp_rate[1, 0],
                         '%.01f' % fp_rate[2, 0],
                         '%.01f' % fp_rate[3, 0],
                         '%.01f' % fp_rate[4, 0],
                         '%.01f' % fp_rate[5, 0]])

    print(table_sizes.get_string(title="Detections based on sizes"))

    # Detection sizes numbers within danger zone
    table_sizes_danger.field_names = ['tiny', 'very small', 'small', 'medium', 'large', 'very large']
    table_sizes_danger.add_row(['%.01f' % tp_rate[0, 1],
                                '%.01f' % tp_rate[1, 1],
                                '%.01f' % tp_rate[2, 1],
                                '%.01f' % tp_rate[3, 1],
                                '%.01f' % tp_rate[4, 1],
                                '%.01f' % tp_rate[5, 1]])

    table_sizes_danger.add_row(['%.01f' % fp_rate[0, 1],
                                '%.01f' % fp_rate[1, 1],
                                '%.01f' % fp_rate[2, 1],
                                '%.01f' % fp_rate[3, 1],
                                '%.01f' % fp_rate[4, 1],
                                '%.01f' % fp_rate[5, 1]])

    print(table_sizes_danger.get_string(title="Detections within danger zone based on sizes"))

    # Ratios between detections within danger zone and whole screen
    table_ratios.field_names = ['tiny', 'very small', 'small', 'medium', 'large', 'very large']
    table_ratios.add_row(['%.01f' % ratios[0, 0],
                          '%.01f' % ratios[1, 0],
                          '%.01f' % ratios[2, 0],
                          '%.01f' % ratios[3, 0],
                          '%.01f' % ratios[4, 0],
                          '%.01f' % ratios[5, 0]])

    table_ratios.add_row(['%.01f' % ratios[0, 1],
                          '%.01f' % ratios[1, 1],
                          '%.01f' % ratios[2, 1],
                          '%.01f' % ratios[3, 1],
                          '%.01f' % ratios[4, 1],
                          '%.01f' % ratios[5, 1]])

    print(table_ratios.get_string(title="Rations between detections within danger zone and all screen"))

    #print(det_sizes_danger)

    plt.show()


# Function parses through the list of detections and checks to which type class it belongs
def update_detection_by_types(det_list, type_index, det_types, cfg):
    # type_index: 0 = TP, 1 = FN
    num_detections = len(det_list)
    for i in range(num_detections):
        det_type = det_list[i]['type']
        if det_type.lower() == cfg.ANALYSIS.OBSTACLE_TYPE_CLASSES[0].lower():
            det_types[0, type_index] += 1
        elif det_type.lower() == cfg.ANALYSIS.OBSTACLE_TYPE_CLASSES[1].lower():
            det_types[1, type_index] += 1
        else:
            det_types[2, type_index] += 1

    return det_types


# Function parses through the list of detections and checks in which size class does the detection fall into
def update_detection_by_sizes(det_list, type_index, det_sizes, cfg):
    # type_index: 0 = TP, 1 = FP, 2 = FN
    num_detections = len(det_list)
    for i in range(num_detections):
        # get detection size
        if type_index == 1:
            det_area = det_list[i]['area']
        else:
            tmp_bb = det_list[i]['bbox']
            det_area = (tmp_bb[3] - tmp_bb[1]) * (tmp_bb[2] - tmp_bb[0])

        # check to which size class it belongs

        # if it is smaller or equal than the smallest size
        if det_area < cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES[0]:
            if type_index == 1:
                det_sizes[0, type_index] += det_list[i]['num_triggers']
            else:
                det_sizes[0, type_index] += 1

        # if it is larger than the largest size
        if det_area >= cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES[-1]:
            if type_index == 1:
                det_sizes[-1, type_index] += det_list[i]['num_triggers']
            else:
                det_sizes[-1, type_index] += 1

        # if it is in-between
        for j in range(1, len(cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES)):
            if cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES[j - 1] <= det_area < cfg.ANALYSIS.OBSTACLE_SIZE_CLASSES[j]:
                if type_index == 1:
                    det_sizes[j, type_index] += det_list[i]['num_triggers']
                else:
                    det_sizes[j, type_index] += 1

    return det_sizes


if __name__ == '__main__':
    main()
