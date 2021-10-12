import numpy as np
import argparse
import os
import sys
import cv2
import time
#import matplotlib.pyplot as plt
from prettytable import PrettyTable
from colorama import Fore, Back, Style
from colorama import init
from datetime import datetime
from utils import *
from core import *
from scipy.stats import norm
from skimage import measure
from pathlib import Path

from configs import get_cfg

from utils.utils import TqdmPool
import utils.context as ctx
from tqdm.auto import tqdm


def get_arguments():
    """ Parse all the arguments provided from the CLI
    Returns: A list of parsed arguments
    """
    parser = argparse.ArgumentParser(description='MODS - A USV-oriented obstacle segmentation benchmark')
    parser.add_argument("method", type=str,
                        help="<required> Method name. This should be equal to the folder name in which the "
                             "segmentation masks are located")
    parser.add_argument("--sequences", type=int, nargs='+', required=False,
                        help="List of sequences on which the evaluation procedure is performed. Zero = all.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of workers for parallel evaluation.")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Config file to use in evaluation. If not specified, the default config is used.")

    return parser.parse_args()


class SequenceEvaluator:
    def __init__(self, gt, gt_coverage, cfg, method_name, sequences, mapping_dict_seq):
        """
        Initialization of Sequence Evaluator
        :param gt: (json) ground truth data
        :param gt_coverage: (json) ground truth obstacle coverage
        :param cfg: (config) config parameters for the evaluation (read from configs/config.py)
        :param sequences: (list) list of sequences on which we perform the evaluation
        :param mapping_dict_seq: (dict) mappings for sequence names
        """

        # Method name
        self.method_name = method_name

        # Ground truths
        self.gt          = gt
        self.gt_coverage = gt_coverage

        # Evaluation configs
        self.cfg         = cfg
        self.sequences   = sequences

        # Sequence name mappings
        self.mapping_dict_seq = mapping_dict_seq

    def process_sequence(self, seq_index_counter):
        """
        Process a single given sequence.
        Note: Implemented so that it can be done in parallel.
        :param seq_index_counter: (int) index of a sequence given in the provided sequences list
        :return:
        """

        # Get evaluation parameters
        eval_params = {"min_overlap":    self.cfg.ANALYSIS.MIN_OVERLAP,
                       "area_threshold": self.cfg.ANALYSIS.AREA_THRESHOLD,
                       "expand_land":    self.cfg.ANALYSIS.EXPAND_LAND,
                       "expand_objs":    self.cfg.ANALYSIS.EXPAND_OBJECTS}

        #gt_coverage = self.gt_coverage

        # Get actual ID of a sequence
        seq_id = self.sequences[seq_index_counter]

        # Get the number of frames in the sequence
        num_frames = self.gt['dataset']['sequences'][seq_id - 1]['num_frames']

        # Get calibration file for the given sequence
        calib_id   = (self.gt['dataset']['sequences'][seq_id - 1]['path'].rstrip().split('/')[1]).split('-')[0]
        calib_file = os.path.join(self.cfg.PATHS.DATASET_CALIB, 'calibration-%s.yaml' % calib_id)
        #calib_file = 'E:/MODB/calibration-%s.yaml' % calib_id

        # Read calibration file
        fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        cM = fs.getNode("M1").mat()  # Extract calibration matrix (M)
        cD = fs.getNode("D1").mat()  # Extract distortion coefficients (D)

        # Check if ignore mask is needed
        if 1 <= seq_id <= 3:
            # Read the ignore mask
            ignore_mask = cv2.imread(os.path.join(self.cfg.PATHS.DATASET, 'mask_seq123.png'), cv2.IMREAD_GRAYSCALE)
        else:
            ignore_mask = None

        # Statistics
        total_overlap_percentages   = []
        total_overlap_percentages_d = []
        total_detections            = np.zeros((6, 1), np.int)  # Order: TP, FP, FN, TP_D, FP_D, FN_D
        total_edge_approx           = []  # RMSE
        total_over_under            = np.zeros(2)  # Overshot, Undershot water-edge nmb. of times when it was not exact
        total_land_detections       = np.zeros(2)  # Total number of TP/FN land detections

        results = []

        # Loop through frames in the sequence
        for frame_number in tqdm(range(num_frames), desc='Seq %s' % seq_id, position=ctx.PID+1, leave=False):

            # Get image's filename and corresponding horizon's filename
            img_name       = self.gt['dataset']['sequences'][seq_id - 1]['frames'][frame_number]['image_file_name']
            img_name_split = img_name.split('.')
            hor_name       = '%s.png' % img_name_split[0]

            # Perform evaluation on the current image
            rmse, num_land_detections, ou_mask, tp_list, fp_list, fn_list, num_fps, \
            tp_list_d, fp_list_d, fn_list_d, num_fps_d, \
            overlap_percentages, overlap_percentages_d = run_evaluation_image(self.cfg,
                                                                              self.method_name,
                                                                              self.gt['dataset']['sequences'][seq_id - 1],
                                                                              self.gt_coverage['sequences'][seq_id - 1],
                                                                              frame_number, eval_params,
                                                                              self.mapping_dict_seq, cM, cD,
                                                                              ignore_mask)

            total_overlap_percentages   = total_overlap_percentages   + overlap_percentages
            total_overlap_percentages_d = total_overlap_percentages_d + overlap_percentages_d

            num_oshoot_wateredge = np.sum(ou_mask == 1)
            num_ushoot_wateredge = np.sum(ou_mask == 2)

            # Add current statistics to the evaluation results
            evaluation_results = {"we_rmse": rmse,
                                  "we_o": int(num_oshoot_wateredge),
                                  "we_u": int(num_ushoot_wateredge),
                                  "we_detections": num_land_detections,
                                  "obstacles": {"tp_list": tp_list,
                                                "fp_list": fp_list,
                                                "fn_list": fn_list},
                                  "obstacles_danger": {"tp_list": tp_list_d,
                                                       "fp_list": fp_list_d,
                                                       "fn_list": fn_list_d},
                                  "img_name": img_name,
                                  "hor_name": hor_name
                                  }

            results.append(evaluation_results)

            # Update quick statistics
            total_detections[0] += len(tp_list)
            total_detections[1] += num_fps
            total_detections[2] += len(fn_list)
            total_detections[3] += len(tp_list_d)
            total_detections[4] += num_fps_d
            total_detections[5] += len(fn_list_d)

            total_edge_approx.append(rmse)
            total_over_under[0]   += num_oshoot_wateredge
            total_over_under[1]   += num_ushoot_wateredge
            total_land_detections += num_land_detections

        return seq_id, results, total_detections, total_edge_approx, total_over_under, total_land_detections,\
               total_overlap_percentages, total_overlap_percentages_d


# Run the evaluation procedure on all sequences of the MODB sequences
# The results are stored in a single JSON file
def run_evaluation():
    init()  # initialize colorama
    args = get_arguments()
    cfg  = get_cfg(args)

    # Create output path if it does not exists yet
    Path(os.path.join(cfg.PATHS.RESULTS)).mkdir(parents=True, exist_ok=True)

    eval_params = {
                   "min_overlap":    cfg.ANALYSIS.MIN_OVERLAP,
                   "area_threshold": cfg.ANALYSIS.AREA_THRESHOLD,
                   "expand_land":    cfg.ANALYSIS.EXPAND_LAND,
                   "expand_objs":    cfg.ANALYSIS.EXPAND_OBJECTS
                  }

    # Read ground truth annotations JSON file
    gt          = read_gt_file(os.path.join(os.path.normpath(cfg.PATHS.DATASET), 'modd3.json')) # 5.10.2021 change; Previously: modb.json
    gt_coverage = read_gt_file(os.path.join(os.path.normpath(cfg.PATHS.DATASET), 'dextr_coverage.json'), True)

    # List of sequences on which we will evaluate the method
    if args.sequences is None:
        sequences = np.arange(1, cfg.DATASET.NUM_SEQUENCES + 1)
    else:
        sequences = args.sequences

    # Get current date and time
    date_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    # Initialize evaluation results dict
    evaluation_results = {
                          "method_name": args.method,
                          "date_time":   date_time,
                          "parameters":  {
                                           "min_overlap": "%0.2f" % cfg.ANALYSIS.MIN_OVERLAP,
                                           "area_threshold": "%d" % cfg.ANALYSIS.AREA_THRESHOLD
                                         },
                          "sequences":   build_sequences_list(cfg.DATASET.NUM_SEQUENCES),
                         }

    # Initialize overlap results dict
    overlap_results = {
                       "method_name":      args.method,
                       "date_time":        date_time,
                       "overlap_perc_all": [],
                       "overlap_perc_dng": []
    }

    # Quick statistics
    # Displaying detection statistics (TP, FP, FN in and out of danger zone) and water-edge statistics
    total_detections = np.zeros((6, 1), np.int)  # TP, FP, FN, TP_D, FP_D, FN_D

    total_edge_approx     = []           # RMSE
    total_over_under      = np.zeros(2)  # Overshot, undershot water-edge (number of times when it was not exact)
    total_land_detections = np.zeros(2)  # Total number of TP/FN land detections

    # Quick statistics that show all overlaps with ground truth
    # This is useful information for selecting the thresholds...
    total_overlap_percentages   = []
    total_overlap_percentages_d = []

    # Build name mapping dict for the current sequence
    mapping_dict_seq = build_mapping_dict(cfg.PATHS.DATASET)

    # Initialize evaluator
    evaluator = SequenceEvaluator(gt, gt_coverage, cfg, args.method, sequences, mapping_dict_seq)

    # Create workers
    with TqdmPool(args.workers) as p:
        eval_generator = p.imap_unordered(evaluator.process_sequence, range(len(sequences)))

        # Go through sequences
        for i, result in tqdm(enumerate(eval_generator), total=len(sequences), desc='Processing sequences'):
            seq_id, results, td, tea, tou, tld, top, topd = result

            evaluation_results['sequences'][seq_id - 1]['frames']    = results
            evaluation_results['sequences'][seq_id - 1]['evaluated'] = True

            total_detections += td
            total_edge_approx.extend(tea)
            total_over_under += tou
            total_land_detections += tld
            total_overlap_percentages.extend(top)
            total_overlap_percentages_d.extend(topd)

    overlap_results['overlap_perc_all'] = total_overlap_percentages
    overlap_results['overlap_perc_dng'] = total_overlap_percentages_d

    # Print quick statistics
    table = PrettyTable()

    tmp_edge = np.ceil(np.mean(total_edge_approx))
    if np.sum(total_over_under) > 0:
        tmp_oshot = (total_over_under[0] / (total_over_under[0] + total_over_under[1])) * 100
        tmp_ushot = (total_over_under[1] / (total_over_under[0] + total_over_under[1])) * 100
    else:
        tmp_oshot = 0
        tmp_ushot = 0

    wedge_line = '%d px ' + Fore.LIGHTRED_EX + '(+%.01f%%, ' + Fore.LIGHTYELLOW_EX + '-%.01f%%)' + Fore.WHITE
    wedge_line = wedge_line % (tmp_edge, tmp_oshot, tmp_ushot)

    if total_land_detections[0] + total_land_detections[1] > 0:
        water_land_detections = '%.01f%%' % ((total_land_detections[0] / (total_land_detections[0] +
                                                                          total_land_detections[1])) * 100)
    else:
        water_land_detections = 100

    tp_line = Fore.LIGHTGREEN_EX + '%d (%d)' + Fore.WHITE
    tp_line = tp_line % (total_detections[0], total_detections[3])

    fp_line = Fore.LIGHTYELLOW_EX + '%d (%d)' + Fore.WHITE
    fp_line = fp_line % (total_detections[1], total_detections[4])

    fn_line = Fore.LIGHTRED_EX + '%d (%d)' + Fore.WHITE
    fn_line = fn_line % (total_detections[2], total_detections[5])

    f1_line = '%.01f%% (%.01f%%)' % ((((2 * total_detections[0]) / (2 * total_detections[0] + total_detections[1] +
                                                                    total_detections[2])) * 100),
                                     (((2 * total_detections[3]) / (2 * total_detections[3] + total_detections[4] +
                                                                    total_detections[5])) * 100))

    table.field_names = ['Water-edge RMSE', 'Water-Land detections', 'TPs', 'FPs', 'FNs', 'F1']
    table.add_row([wedge_line, water_land_detections, tp_line, fp_line, fn_line, f1_line])

    print(table.get_string(title="Results for method %s on %d sequence/s" % (args.method, len(sequences))))

    # Write the evaluation results to JSON file
    write_json_file(cfg.PATHS.RESULTS, args.method, evaluation_results)
    # Write the overlap results to JSON file
    write_json_file(cfg.PATHS.RESULTS, '%s_overlap' % args.method, overlap_results)


# Run evaluation on a single image
def run_evaluation_image(cfg, method_name, gt, gt_coverage, frame_number,
                         eval_params, mapping_dict_seq, M, D, ignore_mask=None):
    """ Function performs evaluation process on a single image
        * In the evaluation process we evaluate the water-edge accuracy and obstacle detection accuracy

    Input params: data_path - path to the MODB raw folder
                  segmentation_path - path to the folder where segmentation masks for all sequences are stored
                  seg_colors - segmentation mask colors (so we can decode them)
                  method_name - name of the method we want to evaluate. The segmentation masks for each sequence should
                                be stored in the parent folder of the same name as method
                  gt - ground truth dictonary
                  frame_number - number of frame which we evaluate
                  eval_param - evaluation parameter values
                  mapping_dict_seq - dictonary with mapping of sequences
                  M - camera calibration matrix
                  D - camera distortion coefficients vector

    Output: - RMSE of the water edge
            - vector of the number of [correctly and incorrectly] assigned water-edge pixels
            - mask where o-shot and u-shot water-edge regions are marked
            - list of true-positives
            - list of false-positives
            - list of false-negatives
            - total number of false-positives (some detections generate more than one false-positive)
            - list of true-positives inside the danger-zone
            - list of false-positives inside the danger zone
            - list of false-negatives inside the danger zone
            - number of false-positives inside the danger zone (some detections generate more than one false-positive)
            - list of overlapping percentages of detections and ground-truth
            - list of overlapping percetanges of detections and ground-truth inside the danger zone
    """

    """ Reading all necessary data... """
    # Read image name:
    img_name       = gt['frames'][frame_number]['image_file_name']
    img_name_split = img_name.split('.')
    seq_path       = gt['path']
    seq_path_split = seq_path.split('/')

    # Look-up name in dict:
    seq_name = mapping_dict_seq[seq_path_split[1]]

    # Read image
    img = cv2.imread(os.path.join(cfg.PATHS.DATASET + seq_path, img_name))

    # Read segmentation mask
    if cfg.SEGMENTATIONS.SEQ_FIRST:
        seg = cv2.imread(os.path.join(cfg.PATHS.SEGMENTATIONS, seq_name,
                                      method_name, "%04d.png" % (frame_number * 10)))
    else:
        seg = cv2.imread(os.path.join(cfg.PATHS.SEGMENTATIONS, method_name, seq_name,
                                      "%04d.png" % (frame_number * 10)))

    # Read horizon mask generated with imu
    horizon_mask = cv2.imread(os.path.join(cfg.PATHS.DATASET, seq_path_split[1], 'imus', '%s.png' % img_name_split[0]),
                              cv2.IMREAD_GRAYSCALE)

    """ Generate danger zone... """
    roll  = gt['frames'][frame_number]['roll']  # Get IMU roll
    pitch = gt['frames'][frame_number]['pitch']  # Get IMU pitch

    # Generate binary danger-zone masks
    danger_zone_mask = danger_zone_to_mask(roll, pitch, cfg.DATASET.CAMERA_HEIGHT, cfg.DATASET.DNG_ZONE_RANGE,
                                           M, D, cfg.DATASET.IMG_WIDTH, cfg.DATASET.IMG_HEIGHT)

    # Code mask to labels
    try:
        seg = code_mask_to_labels(seg, cfg.SEGMENTATIONS.INPUT_COLORS)
    except:
        raise 'Missing number of something for sequence %s frame %d' % (seq_name, frame_number)

    # Resize segmentation mask to match the image
    seg = resize_image(seg, (img.shape[1], img.shape[0]))

    """ Perform the evaluation """
    # Generate obstacle mask
    obstacle_mask = generate_obstacle_mask(seg, gt['frames'][frame_number]['obstacles'])
    # Check if ignore_mask is set (this should be set only for sequences 1, 2 and 3)
    if ignore_mask is not None:
        # Remove all detections that occur in the ignore area
        obstacle_mask = np.logical_and(obstacle_mask, np.logical_not(ignore_mask))

    """ Get connected components of obstacles """
    # Extract connected components from the obstacle mask.
    # The extracted blobs represent potential detections
    obstacle_mask_labels = measure.label(obstacle_mask)
    obstacle_region_list = measure.regionprops(obstacle_mask_labels)
    filtered_region_list = []
    # Delete obstacles that are smaller than threshold
    num_regions = len(obstacle_region_list)
    # Loop through the extracted blobs
    for i in range(num_regions):
        # Check if the surface area is sufficiently large enough
        # If not, then paint over the blob with zero values (aka non-obstacle)
        if obstacle_region_list[i].area < eval_params['area_threshold']:
            # BBOX given in Y_TL, X_TL, Y_BR, X_BR
            obstacle_mask_labels[obstacle_region_list[i].bbox[0]:obstacle_region_list[i].bbox[2],
                                 obstacle_region_list[i].bbox[1]:obstacle_region_list[i].bbox[3]] = 0
            obstacle_mask[obstacle_region_list[i].bbox[0]:obstacle_region_list[i].bbox[2],
                          obstacle_region_list[i].bbox[1]:obstacle_region_list[i].bbox[3]] = 0
        else:
            filtered_region_list.append(obstacle_region_list[i])

    # Perform the evaluation of the water-edge
    rmse, num_land_detections, ou_mask, land_mask, ignore_abv_strad = evaluate_water_edge(gt['frames'][frame_number],
                                                                                          obstacle_mask_labels,
                                                                                          horizon_mask,
                                                                                          eval_params)

    # Calculate GT mask (This is land mask with an extension of the undershot regions)
    gt_mask = (np.logical_or(land_mask, ou_mask == 2)).astype(np.uint8)
    # Expand the mask left and right by 1% of an image width
    gt_mask = expand_land(gt_mask, eval_params)

    # plt.figure(1)
    # plt.subplot(121)
    # plt.imshow(ou_mask)
    # plt.subplot(122)
    # plt.imshow(gt_mask)
    # plt.show()

    """
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(obstacle_mask)
    plt.subplot(222)
    plt.imshow(obstacle_mask_danger)
    plt.subplot(223)
    plt.imshow(gt_mask)
    plt.subplot(224)
    plt.imshow(gt_mask_danger)
    """

    # Read flag if the sequence is exhaustively annotated (aka if all obstacles in the sequence are annotated)
    if gt['exhaustive'] == 1 and gt['frames'][frame_number]['exhaustive'] == 1:
        exhaustive_annotations = True
    else:
        # If not all obstacles in the sequence are annotated, then we shall ignore all false-positive detections,
        # since they may belong to a non-annotated obstacle
        exhaustive_annotations = False

    # Perform the evaluation of the obstacle detection
    tp_list, fp_list, fn_list, num_fps, overlap_percentages,\
    tp_list_d, fp_list_d, fn_list_d, num_fps_d, overlap_perc_d = detect_obstacles_modb(gt['frames'][frame_number],
                                                                                       gt_coverage['frames'][frame_number],
                                                                                       obstacle_mask, gt_mask,
                                                                                       ignore_abv_strad,
                                                                                       horizon_mask, eval_params,
                                                                                       exhaustive_annotations,
                                                                                       danger_zone_mask)

    #plt.show()

    return rmse, num_land_detections, ou_mask, tp_list, fp_list, fn_list, num_fps, tp_list_d, fp_list_d, fn_list_d,\
           num_fps_d, overlap_percentages, overlap_perc_d


if __name__ == '__main__':
    run_evaluation()
